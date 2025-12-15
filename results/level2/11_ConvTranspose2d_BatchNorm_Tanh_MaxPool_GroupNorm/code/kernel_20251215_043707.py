import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_maxpool2d_kernel(
    x_ptr, y_ptr,
    N, C, Hin, Win,
    Hout, Wout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_W: tl.constexpr,
):
    """
    Fused tanh + 2x2 max-pooling (stride=2, padding=0).
    Input:  [N, C, Hin, Win]
    Output: [N, C, Hout, Wout]

    Grid layout (optimized for indexing & fusion):
      pid_n  : batch index      [0 .. N)
      pid_c  : channel index    [0 .. C)
      pid_hw : combined (ho, w-block) index over output

    Each program instance handles:
      - one (n, c, ho) triple
      - a contiguous BLOCK_W-wide vector of Wout positions
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    if (pid_n >= N) | (pid_c >= C):
        return

    # Number of Wout blocks per row
    num_w_blocks = (Wout + BLOCK_W - 1) // BLOCK_W

    ho = pid_hw // num_w_blocks  # scalar
    if ho >= Hout:
        return
    w_block = pid_hw % num_w_blocks

    # Vector of output w indices this program handles
    offs_w = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs_w < Wout  # single, shared mask for all fused ops

    # Map output index -> input top-left corner for 2x2 window (stride=2, no padding)
    h0 = ho * 2
    w0 = offs_w * 2

    # Base pointers for this (n, c, ho)
    base_x = x_ptr + pid_n * stride_xn + pid_c * stride_xc + h0 * stride_xh
    base_y = y_ptr + pid_n * stride_yn + pid_c * stride_yc + ho * stride_yh

    # Pointers for the 2x2 window:
    #   (h0,   w0  )
    #   (h0,   w0+1)
    #   (h0+1, w0  )
    #   (h0+1, w0+1)
    ptr00 = base_x + w0 * stride_xw
    ptr01 = ptr00 + stride_xw
    ptr10 = base_x + stride_xh + w0 * stride_xw
    ptr11 = ptr10 + stride_xw

    # Load inputs (masked on W dimension only; all windows are valid given Hout/Wout definition)
    x00 = tl.load(ptr00, mask=mask, other=0.0)
    x01 = tl.load(ptr01, mask=mask, other=0.0)
    x10 = tl.load(ptr10, mask=mask, other=0.0)
    x11 = tl.load(ptr11, mask=mask, other=0.0)

    # tanh via exp (no tl.tanh available)
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    e2x00 = tl.exp(2.0 * x00)
    t00 = (e2x00 - 1.0) / (e2x00 + 1.0)

    e2x01 = tl.exp(2.0 * x01)
    t01 = (e2x01 - 1.0) / (e2x01 + 1.0)

    e2x10 = tl.exp(2.0 * x10)
    t10 = (e2x10 - 1.0) / (e2x10 + 1.0)

    e2x11 = tl.exp(2.0 * x11)
    t11 = (e2x11 - 1.0) / (e2x11 + 1.0)

    # 2x2 max pooling (all fused ops share same offsets/mask)
    max1 = tl.maximum(t00, t01)
    max2 = tl.maximum(t10, t11)
    pooled = tl.maximum(max1, max2)

    # Store output
    out_ptrs = base_y + offs_w * stride_yw
    tl.store(out_ptrs, pooled, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C, H, W,
    num_groups, group_size,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    """
    GroupNorm over channels and spatial dims for each (N, group).
    Input / Output: [N, C, H, W]
    gamma, beta: [C]

    Optimized layout/indexing:
      - Assume x, y are in contiguous NCHW layout (as produced by PyTorch ConvTranspose2d+BatchNorm).
      - For each (n, g), the group region [group_size, H, W] is contiguous in memory.
      - First pass computes mean/var over this contiguous region.
      - Second pass normalizes and applies affine using the same linear offsets.
    """
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    if (pid_n >= N) | (pid_g >= num_groups):
        return

    HW = H * W
    M = group_size * HW  # elements per group
    c_start = pid_g * group_size

    # Base pointers to the start of this (n, group) block.
    # For contiguous NCHW: stride_xc == H*W, stride_xh == W, stride_xw == 1.
    base_x = x_ptr + pid_n * stride_xn + c_start * stride_xc
    base_y = y_ptr + pid_n * stride_yn + c_start * stride_yc

    offs = tl.arange(0, BLOCK)

    # ---- Pass 1: compute mean and variance over the group ----
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    base = 0
    while base < M:
        idx = base + offs
        mask = idx < M

        # Contiguous access within group: pointer = base_x + idx
        x_ptrs = base_x + idx
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq_val += tl.sum(x_vals * x_vals, axis=0)

        base += BLOCK

    m = sum_val / M
    var = sum_sq_val / M - m * m
    rstd = 1.0 / tl.sqrt(var + eps)

    # ---- Pass 2: normalize and apply affine (fused ops share idx/mask) ----
    base = 0
    while base < M:
        idx = base + offs
        mask = idx < M

        x_ptrs = base_x + idx
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        norm_vals = (x_vals - m) * rstd

        # Channel index for affine parameters
        c_rel = idx // HW
        c = c_start + c_rel

        gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)

        y_vals = norm_vals * gamma + beta

        y_ptrs = base_y + idx
        tl.store(y_ptrs, y_vals, mask=mask)

        base += BLOCK


def tanh_maxpool2d_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Fused tanh + MaxPool2d(kernel_size=2, stride=2, padding=0) using Triton.
    Assumes x is float32, contiguous, on CUDA.
    """
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, Hin, Win = x.shape

    # PyTorch MaxPool2d(kernel_size=2, stride=2, padding=0) output shape
    Hout = (Hin - 2) // 2 + 1
    Wout = (Win - 2) // 2 + 1

    y = torch.empty((N, C, Hout, Wout), device=x.device, dtype=x.dtype)

    BLOCK_W = 128
    num_w_blocks = (Wout + BLOCK_W - 1) // BLOCK_W
    grid = (N, C, Hout * num_w_blocks)

    tanh_maxpool2d_kernel[grid](
        x, y,
        N, C, Hin, Win,
        Hout, Wout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return y


def group_norm_triton(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    GroupNorm implemented in Triton.
    x: [N, C, H, W], float32 CUDA, assumed contiguous.
    weight, bias: [C]
    """
    assert x.is_cuda and x.dtype == torch.float32
    assert weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups

    y = torch.empty_like(x)

    BLOCK = 256  # Larger block for better memory throughput on 4090
    grid = (N, num_groups)

    group_norm_kernel[grid](
        x, y, weight, bias,
        N, C, H, W,
        num_groups, group_size,
        eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) +
    BatchNorm2d (PyTorch native) +
    fused tanh + max-pooling (Triton) +
    GroupNorm (Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        # ConvTranspose2d kept in PyTorch (index mapping is complex)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        # BatchNorm stays in PyTorch to preserve training semantics
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # GroupNorm parameters used by Triton kernel
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        # ConvTranspose2d (PyTorch)
        x = self.conv_transpose(x)
        # BatchNorm2d (PyTorch)
        x = self.batch_norm(x)
        # Fused tanh + maxpool in Triton
        x = tanh_maxpool2d_triton(x)
        # GroupNorm in Triton, using the module's parameters
        x = group_norm_triton(
            x,
            num_groups=self.group_norm.num_groups,
            weight=self.group_norm.weight,
            bias=self.group_norm.bias,
            eps=self.group_norm.eps,
        )
        return x
