# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Higher arithmetic intensity, good when registers allow
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=8,
            num_stages=2,
        ),
        # More rows, fewer cols – better when C_out is small/moderate
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32},
            num_warps=8,
            num_stages=2,
        ),
        # More cols, fewer rows – better when C_out is large
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128},
            num_warps=8,
            num_stages=2,
        ),
        # Conservative tile to keep register pressure lower
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["P", "C_out"],
)
@triton.jit
def conv3d_fwd_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_out, H_out, W_out,
    P,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    inv_div,  # precomputed 1.0 / divisor
    K_D: tl.constexpr,  # kernel depth  (compile-time)
    K_H: tl.constexpr,  # kernel height (compile-time)
    K_W: tl.constexpr,  # kernel width  (compile-time)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile id along flattened output positions
    pid_n = tl.program_id(1)  # tile id along output channels

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

    # Base pointers for all (ic, kd, kh, kw) iterations.
    # Hoist invariants to reduce pointer arithmetic inside inner loops.
    x_base = (
        x_ptr
        + n_idx * stride_xn
        + od_idx * stride_xd
        + oh_idx * stride_xh
        + ow_idx * stride_xw
    )

    w_base = w_ptr + offs_n * stride_woc

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Runtime loop over input channels, with compile-time-unrolled loops
    # over kernel depth/height/width for tighter inner body.
    for ic in range(0, C_in):
        x_ic_base = x_base + ic * stride_xc
        w_ic_base = w_base + ic * stride_wic

        for kd in range(0, K_D):
            x_kd_base = x_ic_base + kd * stride_xd
            w_kd_base = w_ic_base + kd * stride_wkd

            for kh in range(0, K_H):
                x_kh_base = x_kd_base + kh * stride_xh
                w_kh_base = w_kd_base + kh * stride_wkh

                for kw in range(0, K_W):
                    x_ptrs = x_kh_base + kw * stride_xw
                    w_ptrs = w_kh_base + kw * stride_wkw

                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)

                    # Outer product update – all intermediates stay in registers
                    acc += x_vals[:, None] * w_vals[None, :]

    # Add convolution bias (per-output-channel)
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    # Fuse the "divide by divisor" into the convolution to avoid an extra
    # global read+write. inv_div is uniform, so this is very cheap.
    acc *= inv_div

    # Store final result (single store, no intermediate stores)
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
    3D convolution using Triton, matching nn.Conv3d with:
      - stride = 1
      - padding = 0
      - dilation = 1
      - groups = 1

    This kernel fuses:
      - Conv3d (with bias)
      - Division by `divisor`

    Accumulates in fp32 and returns same dtype as input.
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

    # Prepare inverse divisor as a host scalar
    if isinstance(divisor, torch.Tensor):
        inv_div = float(1.0 / divisor.item())
    else:
        inv_div = float(1.0 / divisor)

    # Accumulate in fp32, then cast back to input dtype
    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(P, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv3d_fwd_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_out, H_out, W_out,
        P,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        inv_div,
        K_D=K_D, K_H=K_H, K_W=K_W,
    )

    # Match input dtype if needed
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original Model.
    Uses a custom Triton kernel for Conv3d (fused with division),
    and PyTorch for the remaining ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # Keep the same modules/params as the original model so state_dicts are compatible.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # Use Triton-based Conv3d fused with division by self.divisor
        x = fused_conv3d(x, self.conv.weight, self.conv.bias, self.divisor)

        # Remaining operations in PyTorch
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = x + self.bias
        x = torch.sum(x, dim=self.sum_dim)
        return x
