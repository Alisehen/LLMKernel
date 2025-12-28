import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced tile, good default
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider M tile (more output positions per block)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider N tile (more output channels per block)
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Small, low-register fallback for high-pressure cases
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["N", "C_out", "H_out", "W_out", "K_total"],
)
@triton.jit
def conv2d_div_leakyrelu_kernel(
    x_ptr,  # *f32 or *f16
    w_ptr,  # *f32 or *f16
    bias_ptr,  # *f32
    y_ptr,  # *f32 or *f16
    inv_div,  # scalar
    negative_slope,  # scalar
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    stride_xn,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_wn,
    stride_wc,
    stride_wk,
    stride_wl,
    stride_yn,
    stride_yc,
    stride_yh,
    stride_yw,
    K_total,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implicit-GEMM conv2d (valid, stride=1, no padding) fused with:
      - bias add
      - division by constant
      - LeakyReLU

    Grid:
      pid_m over M = N * H_out * W_out
      pid_n over N = C_out

    Optimized to reduce register pressure:
      - Smaller BLOCK_* tiles (32/64)
      - No persistent 2D base pointer tiles
      - Recompute cheap masks / offsets instead of storing them
      - Avoid explicit a_fp32 / b_fp32 temporaries
    """
    # ----------------------------
    # Program IDs
    # ----------------------------
    pid_m = tl.program_id(0)  # flattened output positions
    pid_n = tl.program_id(1)  # output channels

    # ----------------------------
    # Offsets for this program
    # ----------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    M = N * H_out * W_out

    mask_m = offs_m < M         # [BLOCK_M]
    mask_n = offs_n < C_out     # [BLOCK_N]

    # Decode flattened m -> (n_idx, oh_idx, ow_idx)
    tmp = offs_m
    hw_out = H_out * W_out
    n_idx = tmp // hw_out
    tmp = tmp % hw_out
    oh_idx = tmp // W_out
    ow_idx = tmp % W_out

    # ----------------------------
    # Prepare K loop
    # ----------------------------
    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    KS = KERNEL_SIZE
    KS2 = KS * KS

    # Accumulator (always FP32 for numeric stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----------------------------
    # K reduction loop: implicit GEMM
    # ----------------------------
    for k_start in range(0, K_total, BLOCK_K):
        k_offsets = k_start + offs_k  # [BLOCK_K]
        mask_k = k_offsets < K_total  # [BLOCK_K]

        # Map flattened K -> (c_idx, kh_idx, kw_idx)
        c_idx = k_offsets // KS2
        rem = k_offsets % KS2
        kh_idx = rem // KS
        kw_idx = rem % KS

        # ------------------------
        # Input matrix A: [M, K]
        # ------------------------
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + (oh_idx[:, None] + kh_idx[None, :]) * stride_xh
            + (ow_idx[:, None] + kw_idx[None, :]) * stride_xw
            + c_idx[None, :] * stride_xc
        )  # [BLOCK_M, BLOCK_K]

        mask_a = mask_m[:, None] & mask_k[None, :]  # [BLOCK_M, BLOCK_K]
        a = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # ------------------------
        # Weight matrix B: [K, C_out]
        # ------------------------
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wn
            + c_idx[:, None] * stride_wc
            + kh_idx[:, None] * stride_wk
            + kw_idx[:, None] * stride_wl
        )  # [BLOCK_K, BLOCK_N]

        mask_b = mask_k[:, None] & mask_n[None, :]  # [BLOCK_K, BLOCK_N]
        b = tl.load(w_ptrs, mask=mask_b, other=0.0)

        # GEMM accumulate, let Triton handle precision promotion
        acc += tl.dot(a, b, allow_tf32=True)

    # ----------------------------
    # Fused post-ops
    # ----------------------------

    # Bias add: broadcast over M dimension
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += bias_vals[None, :]  # [BLOCK_M, BLOCK_N]

    # Divide by constant (inv_div precomputed on host)
    acc = acc * inv_div

    # LeakyReLU
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    # ----------------------------
    # Store result
    # ----------------------------
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )  # [BLOCK_M, BLOCK_N]

    mask_store = mask_m[:, None] & mask_n[None, :]  # recomputed cheaply
    tl.store(y_ptrs, acc, mask=mask_store)


def conv2d_div_leakyrelu(x, weight, bias, divisor, negative_slope=0.01):
    """
    Compute:
        y = LeakyReLU( conv2d(x, weight, bias) / divisor )

    Constraints:
      - x: NCHW
      - weight: (C_out, C_in, K, K)
      - stride=1, padding=0, dilation=1, groups=1
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C_in, H_in, W_in = x.shape
    C_out, Cw_in, KH, KW = weight.shape
    assert C_in == Cw_in, "Incompatible in_channels between input and weight"
    assert KH == KW, "Only square kernels are supported"
    KS = KH

    # Valid conv: no padding, stride 1
    H_out = H_in - KS + 1
    W_out = W_in - KS + 1
    assert H_out > 0 and W_out > 0, "Kernel larger than input with no padding"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    K_total = C_in * KS * KS
    M = N * H_out * W_out

    # Precompute inverse divisor to avoid division in kernel
    inv_div = 1.0 / float(divisor)
    neg_slope = float(negative_slope)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_div_leakyrelu_kernel[grid](
        x,
        weight,
        bias,
        y,
        inv_div,
        neg_slope,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        K_total,
        KERNEL_SIZE=KS,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized:
        Conv2d -> divide by constant -> LeakyReLU
    (stride=1, padding=0, dilation=1, groups=1)

    Optimized for RTX 4090 with register-pressure-aware tiling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels supported"
            k = kernel_size[0]
        else:
            k = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.divisor = float(divisor)
        self.negative_slope = 0.01

    def forward(self, x):
        return conv2d_div_leakyrelu(
            x, self.weight, self.bias, self.divisor, self.negative_slope
        )
