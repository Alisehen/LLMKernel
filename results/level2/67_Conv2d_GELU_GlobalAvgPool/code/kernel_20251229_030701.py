# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------
# Conv2d + Bias + GELU
# ----------------------


@triton.autotune(
    configs=[
        # Baseline, conservative (good for multi-input fusion)
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # More warps if register pressure allows – improves compute utilization
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        # Skewed tiles – handle shapes where one dim is small / irregular
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, C_in, H, W,
    C_out, KH, KW,
    H_out, W_out,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output matrix [M, N] where:
    #   M = B * H_out * W_out
    #   N = C_out
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode linear index offs_m -> (batch, oy, ox)
    hw_out = H_out * W_out
    bs = offs_m // hw_out
    rem = offs_m - bs * hw_out
    oy = rem // W_out
    ox = rem - oy * W_out

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_hw = KH * KW

    # GEMM-style loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # (BLOCK_K,)
        mask_k = k_idx < K

        # Decode flattened K index -> (in_channel, ky, kx)
        kk_c = k_idx // k_hw
        kk_rem = k_idx - kk_c * k_hw
        ky = kk_rem // KW
        kx = kk_rem - ky * KW

        # Broadcast for input indexing
        ic_b = kk_c[None, :]   # (1, BLOCK_K)
        ky_b = ky[None, :]     # (1, BLOCK_K)
        kx_b = kx[None, :]     # (1, BLOCK_K)

        oy_b = oy[:, None]     # (BLOCK_M, 1)
        ox_b = ox[:, None]     # (BLOCK_M, 1)
        bs_b = bs[:, None]     # (BLOCK_M, 1)

        in_y = oy_b + ky_b     # (BLOCK_M, BLOCK_K)
        in_x = ox_b + kx_b     # (BLOCK_M, BLOCK_K)

        # Input tile [BLOCK_M, BLOCK_K]
        a_ptrs = (
            x_ptr
            + bs_b * stride_xn
            + ic_b * stride_xc
            + in_y * stride_xh
            + in_x * stride_xw
        )
        # Weight tile [BLOCK_K, BLOCK_N]
        b_ptrs = (
            w_ptr
            + k_idx[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32, allow TF32 for tensor cores on 4090
        acc += tl.dot(a, b, allow_tf32=True)

    # ---- FUSED BIAS + GELU ----
    # Bias vector over channels (N dimension)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # (BLOCK_N,)
    x = acc + bias[None, :]  # (BLOCK_M, BLOCK_N)

    # GELU approximation via tanh implemented manually (no tl.tanh)
    # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    x_sq = x * x
    x_cub = x_sq * x
    inner = x + 0.044715 * x_cub
    inner = 0.7978845608028654 * inner  # sqrt(2/pi)

    exp_arg = 2.0 * inner
    exp_val = tl.exp(exp_arg)
    tanh_inner = (exp_val - 1.0) / (exp_val + 1.0)
    acc = 0.5 * x * (1.0 + tanh_inner)

    # ---- STORE OUTPUT ----
    # Single final store: [B, C_out, H_out, W_out]
    y_ptrs = (
        y_ptr
        + bs[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oy[:, None] * stride_yh
        + ox[:, None] * stride_yw
    )

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv2d_gelu_triton(x, weight, bias):
    """
    x: [B, C_in, H, W]
    weight: [C_out, C_in, KH, KW]
    bias: [C_out]
    returns: [B, C_out, H_out, W_out]
    """
    B, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    H_out = H - KH + 1
    W_out = W - KW + 1

    K = C_in * KH * KW
    M = B * H_out * W_out
    N = C_out

    # Flatten weights to [K, N] with N contiguous for coalesced loads
    w_flat = weight.view(C_out, -1).transpose(0, 1).contiguous()

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    conv2d_gelu_kernel[grid](
        x, w_flat, bias, y,
        B, C_in, H, W,
        C_out, KH, KW,
        H_out, W_out,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_flat.stride(0), w_flat.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


# ----------------------
# Global Average Pool2d
# ----------------------


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def global_avg_pool2d_kernel(
    x_ptr, y_ptr,
    B, C, H, W, HW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
    BLOCK_HW: tl.constexpr,
):
    # One program per (b, c)
    pid = tl.program_id(0)
    bc = pid
    b = bc // C
    c = bc - b * C

    sum_val = tl.zeros((), dtype=tl.float32)

    # Reduction over spatial dimension
    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        tl.multiple_of(offs_hw, BLOCK_HW)
        mask_hw = offs_hw < HW

        h_idx = offs_hw // W
        w_idx = offs_hw - h_idx * W

        x_ptrs = (
            x_ptr
            + b * stride_xn
            + c * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        vals = tl.load(x_ptrs, mask=mask_hw, other=0.0)
        # Warp-level reduction of this tile
        sum_val += tl.sum(vals, axis=0)

    avg = sum_val / HW

    y_ptrs = y_ptr + b * stride_yn + c * stride_yc
    # Single final store
    tl.store(y_ptrs, avg)


def global_avg_pool2d_triton(x):
    """
    x: [B, C, H, W]
    returns: [B, C]
    """
    B, C, H, W = x.shape
    HW = H * W
    y = torch.empty((B, C), device=x.device, dtype=x.dtype)

    def grid(meta):
        # One program per (b, c) pair; ensure grid > 0
        return (max(1, B * C),)

    global_avg_pool2d_kernel[grid](
        x, y,
        B, C, H, W, HW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1),
    )
    return y


# ----------------------
# Full Model
# ----------------------


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = conv2d_gelu_triton(x, self.weight, self.bias)
        x = global_avg_pool2d_triton(x)
        return x
