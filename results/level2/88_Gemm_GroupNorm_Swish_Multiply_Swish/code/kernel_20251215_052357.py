# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------
# Fused Linear (GEMM) + Bias + GroupNorm + Swish * w + Swish
# -----------------------------------------------------------
@triton.autotune(
    configs=[
        # Tuned for Ada / 4090-class GPUs; small-N (GROUP_SIZE) GEMM
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["B", "K"],
)
@triton.jit
def fused_linear_gn_swish2_kernel(
    x_ptr,          # [B, K]
    w_ptr,          # [C, K] as original nn.Linear.weight (out_features, in_features)
    bias_ptr,       # [C]
    gamma_ptr,      # [C] groupnorm weight
    beta_ptr,       # [C] groupnorm bias
    mul_w_ptr,      # [C] multiply_weight
    y_ptr,          # [B, C] output
    B, C, K,
    stride_xb, stride_xk,
    stride_wk, stride_wc,   # IMPORTANT: (k, c) strides for w_ptr
    stride_yb, stride_yc,
    eps,
    GROUP_SIZE: tl.constexpr,   # C / num_groups; also our BLOCK_N
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes one tile: (BLOCK_M x GROUP_SIZE) for a given (batch-tile, group)
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_offset = pid_g * GROUP_SIZE

    # Offsets in B (rows), C (columns for this group), K (reduction)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = group_offset + tl.arange(0, GROUP_SIZE)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_k, BLOCK_K)
    tl.multiple_of(offs_m, BLOCK_M)

    mask_m = offs_m < B
    mask_n = offs_n < C

    # Accumulator for GEMM result: [BLOCK_M, GROUP_SIZE]
    acc = tl.zeros((BLOCK_M, GROUP_SIZE), dtype=tl.float32)

    # GEMM: (B, K) @ (K, C_group) + bias
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K

        a_ptrs = x_ptr + offs_m[:, None] * stride_xb + k_idx[None, :] * stride_xk
        b_ptrs = w_ptr + k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wc

        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

    # Bias add (broadcast along M)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # ----- GroupNorm over GROUP_SIZE channels for each batch row -----
    # Compute mean and variance across the channel axis (N = GROUP_SIZE)
    group_size_f = float(GROUP_SIZE)

    # mean: [BLOCK_M]
    mean = tl.sum(acc, axis=1) / group_size_f

    # variance: E[x^2] - mean^2
    acc_sq = acc * acc
    mean2 = tl.sum(acc_sq, axis=1) / group_size_f
    var = mean2 - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    mean = mean[:, None]
    inv_std = inv_std[:, None]

    # Normalize
    x_norm = (acc - mean) * inv_std

    # Affine (gamma, beta)
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    x_gn = x_norm * gamma[None, :] + beta[None, :]

    # Swish 1: x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig1 = 1.0 / (1.0 + tl.exp(-x_gn))
    x1 = x_gn * sig1

    # Multiply by weight
    mul_w = tl.load(mul_w_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    x2 = x1 * mul_w[None, :]

    # Swish 2
    sig2 = 1.0 / (1.0 + tl.exp(-x2))
    y = x2 * sig2

    # Final store: [B, C]
    y_ptrs = y_ptr + offs_m[:, None] * stride_yb + offs_n[None, :] * stride_yc
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


# -----------------------------
# Wrapper / Launch function
# -----------------------------
def fused_linear_groupnorm_swish2(
    x,
    weight,
    bias,
    gn_weight,
    gn_bias,
    multiply_weight,
    num_groups,
    eps=1e-5,
):
    """
    Fused operation:
      y = Swish( Swish( GroupNorm( x @ W^T + b ) ) * multiply_weight )

    x: [B, in_features]
    weight: [out_features, in_features]  (nn.Linear.weight)
    bias: [out_features]
    gn_weight, gn_bias: [out_features]   (GroupNorm affine params)
    multiply_weight: [out_features]
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda, "Weight must be on CUDA"

    B, K = x.shape
    out_features = weight.shape[0]
    C = out_features

    assert C % num_groups == 0, "out_features must be divisible by num_groups"
    group_size = C // num_groups

    # Output tensor
    y = torch.empty((B, C), device=x.device, dtype=x.dtype)

    # Grid: (batch-tiles, groups)
    def grid(meta):
        return (
            triton.cdiv(B, meta["BLOCK_M"]),
            num_groups,
        )

    fused_linear_gn_swish2_kernel[grid](
        x,
        weight,
        bias,
        gn_weight,
        gn_bias,
        multiply_weight,
        y,
        B,
        C,
        K,
        x.stride(0),
        x.stride(1),
        # Use original weight layout (C, K) as [c, k], but access it as [k, c]
        weight.stride(1),  # stride_wk: step when k++
        weight.stride(0),  # stride_wc: step when c++
        y.stride(0),
        y.stride(1),
        eps,
        GROUP_SIZE=group_size,
    )

    return y


# -----------------------------
# Module
# -----------------------------
class ModelNew(nn.Module):
    """
    Triton-accelerated module:
    Fused: Linear -> GroupNorm -> Swish -> Multiply -> Swish
    Implemented as a single high-performance Triton kernel with
    exactly one global store (final output) and no intermediate
    global-memory writes.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.gn = nn.GroupNorm(num_groups, out_features, eps=1e-5, affine=True)
        self.num_groups = num_groups
        self.eps = self.gn.eps

        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        return fused_linear_groupnorm_swish2(
            x,
            self.linear.weight,
            self.linear.bias,
            self.gn.weight,
            self.gn.bias,
            self.multiply_weight,
            self.num_groups,
            self.eps,
        )
