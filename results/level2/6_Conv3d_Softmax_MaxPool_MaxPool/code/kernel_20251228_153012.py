import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# ---------------------------------------
# 3D Convolution (N, C_in, D, H, W) -> (N, C_out, D_out, H_out, W_out)
# Stride=1, padding=0, dilation=1 (matches default nn.Conv3d)
# ---------------------------------------
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    Kd, Kh, Kw,
    D_out, H_out, W_out,
    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    BLOCK_CO: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over output positions (N * D_out * H_out * W_out)
    pid_co = tl.program_id(1)  # over output channels tiles

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_out

    # Decode pid_m -> (n, d_out, h_out, w_out)
    total_spatial = D_out * H_out * W_out
    n = pid_m // total_spatial
    rem = pid_m % total_spatial

    hw = H_out * W_out
    d_out = rem // hw
    rem2 = rem % hw
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Base offsets for input and output (spatial position fixed)
    base_in = (
        n * stride_in_n
        + d_out * stride_in_d
        + h_out * stride_in_h
        + w_out * stride_in_w
    )

    base_out = (
        n * stride_out_n
        + d_out * stride_out_d
        + h_out * stride_out_h
        + w_out * stride_out_w
    )

    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    bias_vals = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)
    acc += bias_vals

    # Convolution: sum over Cin, Kd, Kh, Kw
    for ci in range(0, C_in):
        for kd in range(0, Kd):
            for kh in range(0, Kh):
                for kw in range(0, Kw):
                    # Single input value for this (ci, kd, kh, kw)
                    in_offset = (
                        base_in
                        + ci * stride_in_c
                        + kd * stride_in_d
                        + kh * stride_in_h
                        + kw * stride_in_w
                    )
                    x_val = tl.load(x_ptr + in_offset)  # scalar

                    # Corresponding weight vector over BLOCK_CO output channels
                    w_offset = (
                        offs_co * stride_w_co
                        + ci * stride_w_ci
                        + kd * stride_w_kd
                        + kh * stride_w_kh
                        + kw * stride_w_kw
                    )
                    w_vals = tl.load(
                        w_ptr + w_offset,
                        mask=mask_co,
                        other=0.0
                    )
                    acc += x_val * w_vals

    # Store result
    out_offset = base_out + offs_co * stride_out_c
    tl.store(
        y_ptr + out_offset,
        acc,
        mask=mask_co,
    )


def conv3d_triton(x, weight, bias):
    """
    x: (N, C_in, D_in, H_in, W_in)
    weight: (C_out, C_in, Kd, Kh, Kw)
    bias: (C_out,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, Cw_in, Kd, Kh, Kw = weight.shape
    assert C_in == Cw_in, "Input channels mismatch with weight"

    # stride=1, padding=0
    D_out = D_in - Kd + 1
    H_out = H_in - Kh + 1
    W_out = W_in - Kw + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw = weight.stride()
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w = y.stride()

    BLOCK_CO = 32  # power-of-two, covers typical C_out (e.g., 16) with masking

    total_pos = N * D_out * H_out * W_out
    grid = lambda META: (
        total_pos,
        triton.cdiv(C_out, META["BLOCK_CO"]),
    )

    conv3d_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
        stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
        stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
        BLOCK_CO=BLOCK_CO,
    )
    return y


# ---------------------------------------
# Softmax over channel dimension (dim=1) for NCDHW
# ---------------------------------------
@triton.jit
def softmax_channel_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    row_id = tl.program_id(0)  # over N * D * H * W

    # Decode row_id -> (n, d, h, w)
    total_spatial = D * H * W
    n = row_id // total_spatial
    rem = row_id % total_spatial

    hw = H * W
    d = rem // hw
    rem2 = rem % hw
    h = rem2 // W
    w = rem2 % W

    base_offset = (
        n * stride_n
        + d * stride_d
        + h * stride_h
        + w * stride_w
    )

    offs_c = tl.arange(0, BLOCK_C)
    c_idx = offs_c
    mask = c_idx < C

    x_ptrs = x_ptr + base_offset + c_idx * stride_c
    x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    # Numerically stable softmax
    x_max = tl.max(x_vals, axis=0)
    x_shifted = x_vals - x_max
    exp_vals = tl.exp(x_shifted)
    denom = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / denom

    tl.store(
        y_ptr + base_offset + c_idx * stride_c,
        softmax_vals,
        mask=mask,
    )


def softmax_channel_triton(x):
    """
    x: (N, C, D, H, W), softmax over dim=1 (channels)
    """
    assert x.is_cuda
    N, C, D, H, W = x.shape
    y = torch.empty_like(x)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    BLOCK_C = 32  # supports typical small C via masking

    rows = N * D * H * W
    grid = lambda META: (rows,)

    softmax_channel_kernel[grid](
        x, y,
        N, C, D, H, W,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        BLOCK_C=BLOCK_C,
    )
    return y


# ---------------------------------------
# 3D MaxPool (kernel_size, stride = kernel_size), no padding
# ---------------------------------------
@triton.jit
def maxpool3d_kernel(
    x_ptr, y_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    Kd, Kh, Kw,
    stride_pool_d, stride_pool_h, stride_pool_w,
    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    numel = N * C * D_out * H_out * W_out
    mask = offs < numel

    # Decode offs -> (n, c, d_out, h_out, w_out)
    w_out = offs % W_out
    tmp = offs // W_out

    h_out = tmp % H_out
    tmp = tmp // H_out

    d_out = tmp % D_out
    tmp = tmp // D_out

    c = tmp % C
    n = tmp // C

    # Base offset of top-left corner of pooling window
    base_in = (
        n * stride_in_n
        + c * stride_in_c
        + (d_out * stride_pool_d) * stride_in_d
        + (h_out * stride_pool_h) * stride_in_h
        + (w_out * stride_pool_w) * stride_in_w
    )

    # Initialize max with very low value
    max_vals = tl.zeros((BLOCK,), dtype=tl.float32) + (-1.0e30)

    for kd in range(0, Kd):
        for kh in range(0, Kh):
            for kw in range(0, Kw):
                in_offset = (
                    base_in
                    + kd * stride_in_d
                    + kh * stride_in_h
                    + kw * stride_in_w
                )
                vals = tl.load(
                    x_ptr + in_offset,
                    mask=mask,
                    other=-1.0e30,
                )
                max_vals = tl.maximum(max_vals, vals)

    out_offset = (
        n * stride_out_n
        + c * stride_out_c
        + d_out * stride_out_d
        + h_out * stride_out_h
        + w_out * stride_out_w
    )

    tl.store(
        y_ptr + out_offset,
        max_vals,
        mask=mask,
    )


def maxpool3d_triton(x, kernel_size):
    """
    x: (N, C, D_in, H_in, W_in)
    kernel_size: int (Kd = Kh = Kw), stride = kernel_size
    """
    assert x.is_cuda
    N, C, D_in, H_in, W_in = x.shape
    Kd = Kh = Kw = int(kernel_size)
    stride_pool_d = stride_pool_h = stride_pool_w = kernel_size

    D_out = (D_in - Kd) // stride_pool_d + 1
    H_out = (H_in - Kh) // stride_pool_h + 1
    W_out = (W_in - Kw) // stride_pool_w + 1

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w = x.stride()
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w = y.stride()

    BLOCK = 128
    numel = N * C * D_out * H_out * W_out
    grid = lambda META: (triton.cdiv(numel, META["BLOCK"]),)

    maxpool3d_kernel[grid](
        x, y,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        Kd, Kh, Kw,
        stride_pool_d, stride_pool_h, stride_pool_w,
        stride_in_n, stride_in_c, stride_in_d, stride_in_h, stride_in_w,
        stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
        BLOCK=BLOCK,
    )
    return y


# ---------------------------------------
# Model using Triton kernels
# ---------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax over channels,
    and performs two 3D max pooling operations, all via Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        k = int(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k, k, k)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.pool_kernel_size = int(pool_kernel_size)

        # Initialize similar to nn.Conv3d default (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * k * k * k
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        w = self.weight
        b = self.bias
        y = conv3d_triton(x, w, b)
        y = softmax_channel_triton(y)
        y = maxpool3d_triton(y, self.pool_kernel_size)
        y = maxpool3d_triton(y, self.pool_kernel_size)
        return y
