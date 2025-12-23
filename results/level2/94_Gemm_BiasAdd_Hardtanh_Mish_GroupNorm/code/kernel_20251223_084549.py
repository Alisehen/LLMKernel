import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Fused GEMM + BiasAdd + Hardtanh + Mish
# ------------------------------------------------------------

@triton.jit
def fused_gemm_bias_hardtanh_mish_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    min_val, max_val,  # Hardtanh limits
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A is (M, K) with strides (stride_am, stride_ak)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B is logically (K, N) = W^T, where W is stored as (N, K).
    # W[n, k] has strides (stride_bn, stride_bk), so:
    #   B[k, n] = W[n, k]
    # => address = n * stride_bn + k * stride_bk
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk  # advance along K-dimension in W (its second dim)

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Hardtanh: clamp between min_val and max_val
    acc = tl.maximum(acc, min_val)
    acc = tl.minimum(acc, max_val)

    # Mish: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    two_sp = 2.0 * sp
    exp_two_sp = tl.exp(two_sp)
    tanh_sp = (exp_two_sp - 1.0) / (exp_two_sp + 1.0)
    acc = acc * tanh_sp

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_gemm_bias_hardtanh_mish(x, weight, bias, min_val=-1.0, max_val=1.0):
    """
    x: (B, in_features)
    weight: (out_features, in_features)  -- stored as (N, K)
    bias: (out_features,)
    returns: (B, out_features)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, K = x.shape
    out_features, in_features = weight.shape
    assert in_features == K
    N = out_features

    x_contig = x.contiguous()
    y = torch.empty((B, N), device=x.device, dtype=x.dtype)

    # We logically want: y = x @ weight.T
    # Treat B = weight.T with shape (K, N), where:
    #   B[k, n] = weight[n, k]
    # For contiguous weight (N, K), strides are:
    #   weight.stride(0) = K (row stride, over n)
    #   weight.stride(1) = 1 (col stride, over k)
    # So to index B[k, n] = weight[n, k], use:
    #   stride_bk = weight.stride(1)  (over k)
    #   stride_bn = weight.stride(0)  (over n)
    stride_bk = weight.stride(1)
    stride_bn = weight.stride(0)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    fused_gemm_bias_hardtanh_mish_kernel[grid](
        x_contig,
        weight,
        bias,
        y,
        B, N, K,
        x_contig.stride(0), x_contig.stride(1),
        stride_bk, stride_bn,
        y.stride(0), y.stride(1),
        min_val, max_val,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return y


# ------------------------------------------------------------
# GroupNorm over (B, C) tensor
# ------------------------------------------------------------

@triton.jit
def groupnorm_forward_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    B, C, G, group_size, eps,
    stride_xb, stride_xc, stride_yb, stride_yc,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index

    offs = tl.arange(0, BLOCK_SIZE)
    group_start = pid_g * group_size
    c_idxs = group_start + offs

    mask = (
        (pid_b < B)
        & (c_idxs < (group_start + group_size))
        & (c_idxs < C)
    )

    x_ptrs = x_ptr + pid_b * stride_xb + c_idxs * stride_xc
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Mean over the group
    mean = tl.sum(x, axis=0) / group_size

    # Variance over the group
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / group_size
    rstd = 1.0 / tl.sqrt(var + eps)

    x_hat = diff * rstd

    gamma = tl.load(gamma_ptr + c_idxs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + c_idxs, mask=mask, other=0.0)

    y = x_hat * gamma + beta

    y_ptrs = y_ptr + pid_b * stride_yb + c_idxs * stride_yc
    tl.store(y_ptrs, y, mask=mask)


def groupnorm_forward(x, weight, bias, num_groups, eps=1e-5):
    """
    x: (B, C)
    weight, bias: (C,)
    GroupNorm over channel dimension with num_groups groups.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x_contig = x.contiguous()
    B, C = x_contig.shape
    assert C == weight.numel() == bias.numel()
    assert C % num_groups == 0
    group_size = C // num_groups

    # Choose block size based on group size, capped at 256
    if group_size <= 16:
        BLOCK_SIZE = 16
    elif group_size <= 32:
        BLOCK_SIZE = 32
    elif group_size <= 64:
        BLOCK_SIZE = 64
    elif group_size <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256

    if group_size > BLOCK_SIZE:
        # Fallback for very large groups; keeps correctness
        return torch.nn.functional.group_norm(
            x, num_groups, weight=weight, bias=bias, eps=eps
        )

    y = torch.empty_like(x_contig)

    grid = lambda META: (B, num_groups)

    groupnorm_forward_kernel[grid](
        x_contig, weight, bias, y,
        B, C, num_groups, group_size, eps,
        x_contig.stride(0), x_contig.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ------------------------------------------------------------
# ModelNew using Triton kernels
# ------------------------------------------------------------

class ModelNew(nn.Module):
    """
    GEMM + BiasAdd + Hardtanh + Mish + GroupNorm
    implemented with high-performance Triton kernels.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        assert bias_shape == (out_features,)

        # Linear layer parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # GroupNorm affine parameters (per-channel scale and shift)
        self.num_groups = num_groups
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))
        self.eps = 1e-5

    def forward(self, x):
        # Move to CUDA if not already (benchmark harness usually does this)
        if not x.is_cuda:
            x = x.cuda()
        if not self.weight.is_cuda:
            self.weight.data = self.weight.data.cuda()
        if not self.bias.is_cuda:
            self.bias.data = self.bias.data.cuda()
        if not self.gn_weight.is_cuda:
            self.gn_weight.data = self.gn_weight.data.cuda()
        if not self.gn_bias.is_cuda:
            self.gn_bias.data = self.gn_bias.data.cuda()

        # Fused GEMM + Bias + Hardtanh + Mish without explicit weight transpose
        x = fused_gemm_bias_hardtanh_mish(x, self.weight, self.bias)

        # GroupNorm over (batch, channels)
        x = groupnorm_forward(x, self.gn_weight, self.gn_bias, self.num_groups, eps=self.eps)
        return x
