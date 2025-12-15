# Triton implementation for:
# GEMM -> GroupNorm -> min(dim=1, keepdim=True) -> bias add (broadcast)

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: C = A @ W^T + bias
    A: [M, K]
    W: [N, K]  (we treat it as [K, N] via strides)
    bias: [N]
    C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Treat w_ptr as matrix of shape [K, N] with strides (stride_wk, stride_wn)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask.T & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, w, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Add linear bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def groupnorm_kernel(
    x_ptr,        # [B, C]
    gamma_ptr,    # [C]
    beta_ptr,     # [C]
    y_ptr,        # [B, C]
    B, C, G,
    channels_per_group,
    stride_xb, stride_xc,
    stride_yb, stride_yc,
    stride_gamma, stride_beta,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GroupNorm over channels for 2D input [B, C].

    For each sample b and group g:
      - group size = channels_per_group = C // G
      - normalize x[b, g*group:(g+1)*group] using mean/var over that group
      - apply affine: y = x_hat * gamma + beta
    """
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G

    valid_row = b < B

    offs = tl.arange(0, BLOCK_SIZE)
    mask_channels = offs < channels_per_group
    c_idx = g * channels_per_group + offs

    mask = valid_row & mask_channels & (c_idx < C)

    x = tl.load(
        x_ptr + b * stride_xb + c_idx * stride_xc,
        mask=mask,
        other=0.0,
    )

    # Compute mean and variance via E[x] and E[x^2]
    group_size = channels_per_group
    mean = tl.sum(x, axis=0) / group_size
    mean2 = tl.sum(x * x, axis=0) / group_size
    var = mean2 - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    x_hat = (x - mean) * inv_std

    gamma = tl.load(
        gamma_ptr + c_idx * stride_gamma,
        mask=mask_channels & (c_idx < C),
        other=1.0,
    )
    beta = tl.load(
        beta_ptr + c_idx * stride_beta,
        mask=mask_channels & (c_idx < C),
        other=0.0,
    )

    y = x_hat * gamma + beta

    tl.store(
        y_ptr + b * stride_yb + c_idx * stride_yc,
        y,
        mask=mask,
    )


@triton.jit
def row_min_kernel(
    x_ptr,       # [B, C]
    out_ptr,     # [B]
    B, C,
    stride_xb, stride_xc,
    stride_out,
    BLOCK_C: tl.constexpr,
):
    """
    Compute per-row minimum over channels:
      out[b] = min_c x[b, c]
    """
    b = tl.program_id(0)
    row_valid = b < B

    offs_c = tl.arange(0, BLOCK_C)
    large = 1e30
    curr_min = tl.full((BLOCK_C,), large, dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        idx_c = c_start + offs_c
        mask = row_valid & (idx_c < C)
        x = tl.load(
            x_ptr + b * stride_xb + idx_c * stride_xc,
            mask=mask,
            other=large,
        )
        curr_min = tl.minimum(curr_min, x)

    row_min = tl.min(curr_min, axis=0)
    tl.store(out_ptr + b * stride_out, row_min, mask=row_valid)


@triton.jit
def broadcast_add_bias_kernel(
    min_ptr,     # [B]
    bias_ptr,    # [C]
    out_ptr,     # [1, C, B, 1]
    B, C,
    stride_min,
    stride_bias,
    stride_o0, stride_o1, stride_o2, stride_o3,
    BLOCK_C: tl.constexpr, BLOCK_B: tl.constexpr,
):
    """
    Compute outer sum:
      out[0, c, b, 0] = min_vals[b] + bias[c]
    Shapes:
      min_vals: [B]
      bias: [C]
      out: [1, C, B, 1]
    """
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    mask_c = offs_c < C
    mask_b = offs_b < B

    bias_vals = tl.load(
        bias_ptr + offs_c * stride_bias,
        mask=mask_c,
        other=0.0,
    )
    min_vals = tl.load(
        min_ptr + offs_b * stride_min,
        mask=mask_b,
        other=0.0,
    )

    # Outer sum: [BLOCK_C, BLOCK_B]
    res = bias_vals[:, None] + min_vals[None, :]

    out_ptrs = (
        out_ptr
        + offs_c[:, None] * stride_o1
        + offs_b[None, :] * stride_o2
    )

    tl.store(
        out_ptrs,
        res,
        mask=mask_c[:, None] & mask_b[None, :],
    )


def fused_linear_groupnorm_min_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    linear_bias: torch.Tensor,
    gn_weight: torch.Tensor,
    gn_bias: torch.Tensor,
    num_groups: int,
    final_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    x:          [B, in_features]
    weight:     [out_features, in_features]
    linear_bias:[out_features]
    gn_weight:  [out_features]
    gn_bias:    [out_features]
    final_bias: [1, out_features, 1, 1]
    Returns:
      out: [1, out_features, B, 1]
    """
    assert x.is_cuda and weight.is_cuda and final_bias.is_cuda
    B, K = x.shape
    N = weight.shape[0]
    C = N
    assert gn_weight.shape[0] == C
    assert gn_bias.shape[0] == C
    assert C % num_groups == 0
    channels_per_group = C // num_groups

    # GEMM: [B, K] @ [K, N] + linear_bias -> [B, N]
    y = torch.empty((B, N), device=x.device, dtype=torch.float32)

    grid_linear = lambda META: (
        triton.cdiv(B, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    linear_kernel[grid_linear](
        x, weight, linear_bias, y,
        B, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )

    # GroupNorm on [B, C] = [B, N]
    y_norm = torch.empty_like(y)

    grid_gn = (B * num_groups,)
    groupnorm_kernel[grid_gn](
        y, gn_weight, gn_bias, y_norm,
        B, C, num_groups,
        channels_per_group,
        y.stride(0), y.stride(1),
        y_norm.stride(0), y_norm.stride(1),
        gn_weight.stride(0), gn_bias.stride(0),
        eps,
        BLOCK_SIZE=32,  # power-of-2, >= channels_per_group (16 here)
    )

    # Min over channels dim=1: result [B]
    mins = torch.empty((B,), device=x.device, dtype=torch.float32)
    grid_min = (B,)
    row_min_kernel[grid_min](
        y_norm,
        mins,
        B, C,
        y_norm.stride(0), y_norm.stride(1),
        mins.stride(0),
        BLOCK_C=128,
    )

    # Broadcast add with final bias: [1, C, B, 1]
    out = torch.empty((1, C, B, 1), device=x.device, dtype=torch.float32)
    bias_vec = final_bias.view(C)

    grid_bcast = lambda META: (
        triton.cdiv(C, META['BLOCK_C']),
        triton.cdiv(B, META['BLOCK_B']),
    )

    broadcast_add_bias_kernel[grid_bcast](
        mins,
        bias_vec,
        out,
        B, C,
        mins.stride(0),
        bias_vec.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=64,
        BLOCK_B=64,
    )

    return out


class ModelNew(nn.Module):
    """
    Model that performs:
      Linear (GEMM) -> GroupNorm -> min over channels -> bias add (broadcast)
    Implemented with Triton kernels.
    """

    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Keep the same submodules as the original model so that
        # state_dict / weight loading remains compatible.
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_groupnorm_min_bias(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.bias,
            self.group_norm.eps,
        )
