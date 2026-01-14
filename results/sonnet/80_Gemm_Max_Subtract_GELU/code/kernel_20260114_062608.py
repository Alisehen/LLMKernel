import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gemm_max_mean_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize max values to -inf
    max_vals = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    
    # Process all N columns in blocks
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        
        # Compute GEMM for this block
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        for k in range(0, K, BLOCK_K):
            k_remaining = K - k
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b, allow_tf32=True)
            
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        
        # Add bias
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
        
        # Update max values (reduce along N dimension)
        # Mask invalid columns
        col_mask = offs_n[None, :] < N
        acc_masked = tl.where(col_mask, acc, float('-inf'))
        block_max = tl.max(acc_masked, axis=1)
        max_vals = tl.maximum(max_vals, block_max)
    
    # After max along dim=1, we have shape (BLOCK_M,)
    # Mean along dim=1 of a single value is itself
    # So result = max_val - max_val = 0
    result = max_vals - max_vals  # = 0
    
    # GELU(0) = 0
    # GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using tanh approximation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = result * result * result
    inner = sqrt_2_over_pi * (result + coeff * x3)
    
    # Compute tanh manually: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    gelu_result = 0.5 * result * (1.0 + tanh_val)
    
    # Store result (shape: M x 1)
    out_offs = offs_m
    tl.store(c_ptr + out_offs, gelu_result, mask=offs_m < M)


def gemm_max_mean_gelu(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    
    c = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    
    b = weight.t().contiguous()
    
    BLOCK_M = 32
    BLOCK_N = 128
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M),)
    
    gemm_max_mean_gelu_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1), b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        return gemm_max_mean_gelu(x, self.gemm.weight, self.gemm.bias)
