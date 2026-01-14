import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gemm_bias_hardtanh_mish_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matmul loop with unrolling hint
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        mask_a = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_b = (mask_k[:, None]) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused: add bias (broadcast)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Fused: Hardtanh (clamp between -1 and 1)
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Fused: Mish = x * tanh(softplus(x))
    # Recompute softplus to reduce register pressure
    # Use fast path for large values
    mask_large = acc > 20.0
    softplus = tl.where(mask_large, acc, tl.log(1.0 + tl.exp(tl.minimum(acc, 20.0))))
    
    # Fast tanh approximation: (exp(2x) - 1) / (exp(2x) + 1)
    two_softplus = 2.0 * softplus
    exp_2x = tl.exp(tl.minimum(two_softplus, 20.0))
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    result = acc * tanh_val

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=mask_c)


@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    M, N, num_groups,
    stride_xm, stride_xn, stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    
    channels_per_group = N // num_groups
    start_channel = group_idx * channels_per_group
    
    # Single-pass Welford's algorithm for mean and variance
    mean_val = 0.0
    m2_val = 0.0
    count = 0
    
    # First pass: compute mean and variance
    for c_offset in range(0, channels_per_group, BLOCK_SIZE):
        offs = start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < start_channel + channels_per_group
        x_ptrs = x_ptr + batch_idx * stride_xm + offs * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Welford update - vectorized
        for i in range(BLOCK_SIZE):
            is_valid = (c_offset + i) < channels_per_group
            val = tl.where(is_valid, x, 0.0)
            count_cond = tl.where(is_valid, 1, 0)
            count += count_cond
            delta = val - mean_val
            mean_val += tl.where(count_cond > 0, delta / tl.maximum(count, 1), 0.0)
            delta2 = val - mean_val
            m2_val += tl.where(count_cond > 0, delta * delta2, 0.0)
    
    var_val = m2_val / tl.maximum(channels_per_group, 1)
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Second pass: normalize and apply affine transform
    for c_offset in range(0, channels_per_group, BLOCK_SIZE):
        offs = start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < start_channel + channels_per_group
        x_ptrs = x_ptr + batch_idx * stride_xm + offs * stride_xn
        out_ptrs = out_ptr + batch_idx * stride_om + offs * stride_on
        
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
        
        # Fused normalize and affine
        out = (x - mean_val) * rstd * gamma + beta
        
        tl.store(out_ptrs, out, mask=mask)


@triton.jit
def groupnorm_kernel_optimized(
    x_ptr, out_ptr, gamma_ptr, beta_ptr,
    M, N, num_groups,
    stride_xm, stride_xn, stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    batch_idx = pid // num_groups
    group_idx = pid % num_groups
    
    channels_per_group = N // num_groups
    start_channel = group_idx * channels_per_group
    
    # Compute mean
    sum_val = 0.0
    for c_offset in range(0, channels_per_group, BLOCK_SIZE):
        offs = start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < start_channel + channels_per_group
        x_ptrs = x_ptr + batch_idx * stride_xm + offs * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_val += tl.sum(tl.where(mask, x, 0.0))
    
    mean_val = sum_val / channels_per_group
    
    # Compute variance
    sum_sq = 0.0
    for c_offset in range(0, channels_per_group, BLOCK_SIZE):
        offs = start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < start_channel + channels_per_group
        x_ptrs = x_ptr + batch_idx * stride_xm + offs * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        diff = x - mean_val
        sum_sq += tl.sum(tl.where(mask, diff * diff, 0.0))
    
    var_val = sum_sq / channels_per_group
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Normalize and apply affine transform
    for c_offset in range(0, channels_per_group, BLOCK_SIZE):
        offs = start_channel + c_offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < start_channel + channels_per_group
        x_ptrs = x_ptr + batch_idx * stride_xm + offs * stride_xn
        out_ptrs = out_ptr + batch_idx * stride_om + offs * stride_on
        
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
        
        out = (x - mean_val) * rstd * gamma + beta
        
        tl.store(out_ptrs, out, mask=mask)


def fused_gemm_bias_hardtanh_mish(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    # Optimized block sizes for RTX 4090
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    fused_gemm_bias_hardtanh_mish_kernel[grid](
        x, b, bias, c, M, N, K,
        x.stride(0), x.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def groupnorm_triton(x, num_groups, gamma, beta, eps=1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    
    # Optimized block size for better memory access
    BLOCK_SIZE = 256
    
    grid = lambda META: (M * num_groups,)
    groupnorm_kernel_optimized[grid](
        x, out, gamma, beta,
        M, N, num_groups,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = fused_gemm_bias_hardtanh_mish(x, self.weight, self.bias)
        x = groupnorm_triton(x, self.num_groups, self.gamma, self.beta)
        return x
