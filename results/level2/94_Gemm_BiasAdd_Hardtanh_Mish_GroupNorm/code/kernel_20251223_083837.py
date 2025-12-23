# complete ModelNew code with optimized Triton kernels
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_bias_hardtanh_mish_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output matrix C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias: [N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Hardtanh: clamp between -1 and 1
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Mish: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    # Numerically-stable-ish but simple implementation
    softplus = tl.log(1.0 + tl.exp(acc))
    two_sp = 2.0 * softplus
    t = tl.exp(two_sp)
    tanh_sp = (t - 1.0) / (t + 1.0)
    acc = acc * tanh_sp

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear_bias_hardtanh_mish(x, weight, bias):
    """
    x:      [B, I]
    weight: [O, I]
    bias:   [O]
    returns: [B, O]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, I = x.shape
    O = weight.shape[0]

    # Use W^T as [I, O] for GEMM
    w_t = weight.t().contiguous()
    out = torch.empty((B, O), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            max(1, triton.cdiv(B, META['BLOCK_M'])),
            max(1, triton.cdiv(O, META['BLOCK_N'])),
        )

    fused_linear_bias_hardtanh_mish_kernel[grid](
        x, w_t, bias, out,
        B, O, I,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return out


@triton.jit
def groupnorm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, G, group_size, eps,
    stride_xn, stride_xc, stride_yn, stride_yc,
    BLOCK_C: tl.constexpr,
):
    # One program per (batch, group) pair
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    # Optional safety guard if grid is over-provisioned (no loops here, allowed)
    if n >= B:
        return

    offs_c = tl.arange(0, BLOCK_C)

    # First pass: compute mean and variance for this (n, g)
    sum_val = 0.0
    sum_sq = 0.0

    for c0 in range(0, group_size, BLOCK_C):
        rel_c = c0 + offs_c          # [BLOCK_C]
        mask = rel_c < group_size    # [BLOCK_C]
        c_idx = g * group_size + rel_c  # [BLOCK_C] -> channel indices in this group

        x_ptrs = x_ptr + n * stride_xn + c_idx * stride_xc
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Aggregate into scalars
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    # Use group_size directly; Triton handles int->float promotion
    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for c0 in range(0, group_size, BLOCK_C):
        rel_c = c0 + offs_c
        mask = rel_c < group_size
        c_idx = g * group_size + rel_c

        x_ptrs = x_ptr + n * stride_xn + c_idx * stride_xc
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        gamma = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

        x_norm = (x - mean) * inv_std
        y = x_norm * gamma + beta

        y_ptrs = y_ptr + n * stride_yn + c_idx * stride_yc
        tl.store(y_ptrs, y, mask=mask)


def triton_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    """
    x: [B, C]
    weight, bias: [C]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, C = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups

    y = torch.empty_like(x)

    def grid(META):
        # One program per (batch, group) pair
        return (max(1, B * num_groups),)

    groupnorm_kernel[grid](
        x, weight, bias, y,
        B, C, num_groups, group_size, eps,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_C=128,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
        Linear -> BiasAdd -> Hardtanh -> Mish -> GroupNorm
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Match original Linear parameter shapes
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # GroupNorm affine parameters over channels (out_features)
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        # Move to CUDA if not already (to use Triton kernels)
        if not x.is_cuda:
            x = x.cuda()
        if not self.weight.is_cuda:
            self.weight.data = self.weight.data.cuda()
            self.bias.data = self.bias.data.cuda()
            self.gn_weight.data = self.gn_weight.data.cuda()
            self.gn_bias.data = self.gn_bias.data.cuda()

        x = fused_linear_bias_hardtanh_mish(x, self.weight, self.bias)
        x = triton_groupnorm(x, self.gn_weight, self.gn_bias, self.num_groups, self.eps)
        return x
