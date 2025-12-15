# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["P", "C_out"],
)
@triton.jit
def conv3d_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    P,
    divisor,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over P = N*D_out*H_out*W_out
    BLOCK_N: tl.constexpr,  # tile over C_out
):
    # Program ids for (output positions, output channels)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in flattened output position (M) and output channels (N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode flattened output index offs_m -> (n_idx, od_idx, oh_idx, ow_idx)
    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    od_idx = rem // HW_out
    rem2 = rem % HW_out
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    # Base pointers for this tile's output positions and channels
    # x[n, :, od, oh, ow]
    x_base = (
        x_ptr
        + n_idx * stride_xn
        + od_idx * stride_xd
        + oh_idx * stride_xh
        + ow_idx * stride_xw
    )
    # w[oc, :, :, :, :]
    w_base = w_ptr + offs_n * stride_woc

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 3D convolution: iterate over input channels and kernel volume
    # Kernel spatial dimensions are constexpr, so we can unroll them.
    for ic in range(0, C_in):
        ic_offset_x = ic * stride_xc
        ic_offset_w = ic * stride_wic
        for kd in tl.static_range(0, K_D):
            kd_offset_x = kd * stride_xd
            kd_offset_w = kd * stride_wkd
            for kh in tl.static_range(0, K_H):
                kh_offset_x = kh * stride_xh
                kh_offset_w = kh * stride_wkh
                for kw in tl.static_range(0, K_W):
                    kw_offset_x = kw * stride_xw
                    kw_offset_w = kw * stride_wkw

                    # Input pointers for this (ic, kd, kh, kw) slice, across BLOCK_M positions
                    x_ptrs = x_base + ic_offset_x + kd_offset_x + kh_offset_x + kw_offset_x
                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

                    # Weight pointers for this (ic, kd, kh, kw) slice, across BLOCK_N output channels
                    w_ptrs = w_base + ic_offset_w + kd_offset_w + kh_offset_w + kw_offset_w
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)

                    # Outer product update
                    acc += x_vals[:, None] * w_vals[None, :]

    # Add convolution bias (broadcast across BLOCK_M) and fuse scalar division
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]
    acc = acc / divisor

    # Store result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def fused_conv3d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, divisor) -> torch.Tensor:
    """
    High-performance 3D convolution using Triton, matching nn.Conv3d with:
      - stride = 1
      - padding = 0
      - dilation = 1
      - groups = 1

    Fuses:
      - Conv3d
      - Bias add
      - Scalar division by `divisor`
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 5 and weight.ndim == 5

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in_w == C_in, "Groups other than 1 are not supported in this kernel."

    # Output spatial size for stride=1, padding=0, dilation=1
    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert D_out > 0 and H_out > 0 and W_out > 0

    P = N * D_out * H_out * W_out  # total output positions

    y = torch.empty((N, C_out, D_out, H_out, W_out),
                    device=x.device, dtype=torch.float32)

    # Scalar divisor (host-side) for fused division
    if isinstance(divisor, torch.Tensor):
        # Assume scalar tensor; move to host float
        divisor_val = float(divisor.item())
    else:
        divisor_val = float(divisor)

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv3d_fwd_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        P,
        divisor_val,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        K_D=K_D, K_H=K_H, K_W=K_W,
    )

    # Match input dtype if needed
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original Model.

    Uses a custom Triton kernel for Conv3d (+bias + scalar division),
    and PyTorch for the remaining ops:
      - MaxPool3d
      - AdaptiveAvgPool3d
      - Bias add
      - Final reduction (sum)
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # Triton-based Conv3d with fused bias and scalar division
        x = fused_conv3d(x, self.conv.weight, self.conv.bias, self.divisor)

        # Remaining operations in PyTorch (not fused due to reductions / layout changes)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x
