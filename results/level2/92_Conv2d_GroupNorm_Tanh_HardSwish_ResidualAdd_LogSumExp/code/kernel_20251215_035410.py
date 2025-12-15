# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Optimized NCHW Conv2D kernel (stride=1, padding=0, dilation=1)
# ------------------------------------------------------------

@triton.autotune(
    configs=[
        # Larger tile – best when register pressure allows
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=2,
        ),
        # More conservative tile – reduces register pressure
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32},
            num_warps=2,
            num_stages=2,
        ),
        # Narrow OC tile – good when many output channels
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 16},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['N', 'C_in', 'H_in', 'W_in', 'OC', 'KH', 'KW'],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    OC, KH, KW,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # flattened (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # output channels (OC)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < OC

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator: keep in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main convolution loop over (C_in, KH, KW)
    for ic in range(0, C_in):
        for kh in range(0, KH):
            h_in = oh_idx + kh
            for kw in range(0, KW):
                w_in = ow_idx + kw

                # Input load (NCHW, stride=1,pad=0 so accesses are mostly contiguous in W)
                x_ptrs = (
                    x_ptr
                    + n_idx * stride_xn
                    + ic * stride_xc
                    + h_in * stride_xh
                    + w_in * stride_xw
                )
                x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M]

                # Weight load
                w_ptrs = (
                    w_ptr
                    + offs_n * stride_wo
                    + ic * stride_wc
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]

                # Outer product accumulation
                acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += bias_vals[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_triton_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    NCHW convolution with stride=1, padding=0, dilation=1 implemented in Triton.
    x:      [N, C_in, H_in, W_in]
    weight: [OC, C_in, KH, KW]
    bias:   [OC]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 4 and weight.ndim == 4
    N, C_in, H_in, W_in = x.shape
    OC, C_in2, KH, KW = weight.shape
    assert C_in == C_in2, "Input channel mismatch between x and weight"

    # stride=1, padding=0, dilation=1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    assert H_out > 0 and W_out > 0

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    y = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=torch.float32)

    P = N * H_out * W_out

    # Grid is defined as a lambda so autotune configs can change BLOCK_M/BLOCK_N
    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

    conv2d_nchw_kernel[grid](
        x_contig, w_contig, b_contig, y,
        N, C_in, H_in, W_in,
        OC, KH, KW,
        H_out, W_out,
        x_contig.stride(0), x_contig.stride(1),
        x_contig.stride(2), x_contig.stride(3),
        w_contig.stride(0), w_contig.stride(1),
        w_contig.stride(2), w_contig.stride(3),
        y.stride(0), y.stride(1),
        y.stride(2), y.stride(3),
    )

    return y


# ------------------------------------------------------------
# Optimized LogSumExp over channel dim (NCHW, dim=1, keepdim=True)
# Single-pass numerically stable algorithm to halve memory traffic.
# ------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 256},
            num_warps=4,
            num_stages=2,
        ),
        # Smaller tile as conservative fallback if registers get tight
        triton.Config(
            {'BLOCK_M': 128},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def logsumexp_channel_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # flattened (N * H * W)
):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    P = N * H * W
    mask_m = offs_m < P

    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    # Initialize with channel 0
    c0 = 0
    x_ptrs0 = (
        x_ptr
        + n_idx * stride_xn
        + c0 * stride_xc
        + h_idx * stride_xh
        + w_idx * stride_xw
    )
    x0 = tl.load(x_ptrs0, mask=mask_m, other=-float("inf"))

    # Running max (m) and running sum of exp shifted by m (s)
    m = x0
    # exp(x0 - m) = 1, keeps everything in-register without needing tl.ones_like
    s = tl.exp(x0 - m)

    # Single-pass numerically stable log-sum-exp
    for c in range(1, C):
        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + c * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=mask_m, other=-float("inf"))

        new_m = tl.maximum(m, x_vals)
        # s_new = s * exp(m - new_m) + exp(x_vals - new_m)
        s = s * tl.exp(m - new_m) + tl.exp(x_vals - new_m)
        m = new_m

    out = tl.log(s) + m

    # Store result: y shape [N, 1, H, W], channel index is 0
    y_ptrs = (
        y_ptr
        + n_idx * stride_yn
        + h_idx * stride_yh
        + w_idx * stride_yw
    )
    tl.store(y_ptrs, out, mask=mask_m)


def logsumexp_channel_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute torch.logsumexp(x, dim=1, keepdim=True) for NCHW tensor x using Triton.
    x: [N, C, H, W] -> returns [N, 1, H, W]
    """
    assert x.is_cuda
    assert x.ndim == 4
    N, C, H, W = x.shape

    x_contig = x.contiguous()
    y = torch.empty((N, 1, H, W), device=x.device, dtype=torch.float32)

    P = N * H * W

    def grid(meta):
        return (triton.cdiv(P, meta['BLOCK_M']),)

    logsumexp_channel_kernel[grid](
        x_contig, y,
        N, C, H, W,
        x_contig.stride(0), x_contig.stride(1),
        x_contig.stride(2), x_contig.stride(3),
        y.stride(0), y.stride(1),
        y.stride(2), y.stride(3),
    )

    return y


# ------------------------------------------------------------
# ModelNew using optimized Triton kernels
# ------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Triton-accelerated model:
      Conv2d (NCHW, stride=1, padding=0) via Triton
      GroupNorm (PyTorch)
      Tanh (PyTorch)
      HardSwish (PyTorch)
      Residual Add (PyTorch)
      LogSumExp over channels via Triton
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Convolution via Triton
        weight = self.conv.weight
        bias = self.conv.bias
        x_conv = conv2d_triton_nchw(x, weight, bias)

        # 2) Group Normalization
        x_norm = self.group_norm(x_conv)

        # 3) Tanh
        x_tanh = self.tanh(x_norm)

        # 4) HardSwish
        x_hard_swish = self.hard_swish(x_tanh)

        # 5) Residual Addition
        x_res = x_conv + x_hard_swish

        # 6) LogSumExp over channels (dim=1, keepdim=True) via Triton
        x_logsumexp = logsumexp_channel_triton(x_res)

        return x_logsumexp
