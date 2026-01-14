import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gemm_bias_sigmoid_scale_residual_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    scaling_factor,
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
    
    # GEMM loop
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        
        mask_a = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_b = (mask_k[:, None]) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load bias once
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    
    # Add bias (broadcast along M dimension)
    acc += bias[None, :]
    
    # Store original for residual (keep in register)
    original_x = acc
    
    # Fused sigmoid: 1 / (1 + exp(-x))
    # Recompute -acc instead of storing to save registers
    exp_neg = tl.exp(-acc)
    x = 1.0 / (1.0 + exp_neg)
    
    # Fused scaling (cheap mul, recompute if needed)
    x = x * scaling_factor
    
    # Fused residual add
    x = x + original_x

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, x, mask=mask_c)


def fused_gemm_bias_sigmoid_scale_residual(x, weight, bias, scaling_factor):
    M, K = x.shape
    N = weight.shape[0]
    
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    # Autotune configurations with register pressure awareness
    # Start with conservative sizes for matmul fusion
    # Config 1: Balanced for RTX 4090
    # Config 2: Smaller blocks as fallback if register pressure high
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']), 
            triton.cdiv(N, META['BLOCK_N'])
        )
    
    # Use autotune to find best config
    # RTX 4090: 128 KB L1, 48 MB L2, 101376 bytes shared memory
    # Target: maximize occupancy while avoiding register spills
    
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ]
    
    # For now, use the first config (can be extended with @triton.autotune)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    fused_gemm_bias_sigmoid_scale_residual_kernel[grid](
        x, b, bias, c, M, N, K,
        scaling_factor,
        x.stride(0), x.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3
    )
    
    return c


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return fused_gemm_bias_sigmoid_scale_residual(x, self.weight, self.bias, self.scaling_factor)
