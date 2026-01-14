import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gemm_swish_ops_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    has_bias: tl.constexpr,
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
    
    # Matmul loop with better memory access pattern
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Compute mask once
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Add bias if present
    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # Fused operations - optimized to reduce register pressure
    # Swish: x * sigmoid(x) - use efficient sigmoid approximation
    # Clamp input to avoid overflow in exp
    acc_clamped = tl.maximum(tl.minimum(acc, 20.0), -20.0)
    sigmoid_acc = 1.0 / (1.0 + tl.exp(-acc_clamped))
    acc = acc * sigmoid_acc
    
    # Divide by 2.0 and clamp to [-1, 1]
    acc = acc * 0.5
    acc = tl.maximum(tl.minimum(acc, 1.0), -1.0)
    
    # Tanh using efficient approximation for range [-1, 1]
    # tanh(x) ≈ (exp(2x) - 1) / (exp(2x) + 1) for small x
    # For x in [-1, 1], use direct formula
    acc_2x = acc * 2.0
    exp_2x = tl.exp(acc_2x)
    acc = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Final clamp to [-1, 1]
    acc = tl.maximum(tl.minimum(acc, 1.0), -1.0)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_out)


def fused_gemm_swish_ops(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    has_bias = bias is not None
    bias_ptr = bias if has_bias else x
    
    # Optimized block sizes for RTX 4090 with register pressure awareness
    # Analysis: Low SM throughput (43%) and low warp activity (16.66%) suggest
    # we need better occupancy. Reduce block sizes to lower register usage.
    
    # Conservative block sizes to avoid register spilling
    # Target: keep registers_per_thread < 128
    if M >= 2048 and N >= 2048 and K >= 2048:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_warps = 4
        num_stages = 3
    elif M >= 1024 and N >= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_warps = 4
        num_stages = 2
    elif K >= 4096:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        num_warps = 4
        num_stages = 2
    
    grid = (
        triton.cdiv(M, BLOCK_M), 
        triton.cdiv(N, BLOCK_N)
    )
    
    fused_gemm_swish_ops_kernel[grid](
        x, b, bias_ptr, c, M, N, K,
        x.stride(0), x.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x):
        return fused_gemm_swish_ops(x, self.weight, self.bias)
