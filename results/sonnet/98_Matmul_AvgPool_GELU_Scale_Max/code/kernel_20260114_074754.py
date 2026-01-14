import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_matmul_pool_gelu_scale_max_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    pool_kernel_size, scale_factor,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    N_pooled = N // pool_kernel_size
    
    # Initialize max value to very negative number
    max_val = tl.full((BLOCK_M,), -1e10, dtype=tl.float32)
    
    # Process each pooled output sequentially
    for pool_idx in range(N_pooled):
        # Accumulate over pool_kernel_size elements
        pool_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        
        for k_pool in range(pool_kernel_size):
            n_idx = pool_idx * pool_kernel_size + k_pool
            
            # Compute matmul for this specific output column
            acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
            
            for k in range(0, K, BLOCK_K):
                offs_k = k + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                
                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
                w_ptrs = weight_ptr + n_idx * stride_wn + offs_k * stride_wk
                
                x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                w_block = tl.load(w_ptrs, mask=mask_k, other=0.0)
                
                acc += tl.sum(x_block * w_block[None, :], axis=1)
            
            # Add bias
            bias_val = tl.load(bias_ptr + n_idx)
            acc += bias_val
            
            # Accumulate into pool
            pool_acc += acc
        
        # Apply average pooling
        pool_avg = pool_acc / tl.cast(pool_kernel_size, tl.float32)
        
        # Apply GELU
        sqrt_2_over_pi = 0.7978845608028654
        x = pool_avg
        x3 = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x3)
        
        # Clamp to avoid overflow
        inner = tl.where(inner > 10.0, 10.0, inner)
        inner = tl.where(inner < -10.0, -10.0, inner)
        
        exp_2x = tl.exp(2.0 * inner)
        tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
        gelu_val = 0.5 * x * (1.0 + tanh_inner)
        
        # Apply scaling
        scaled_val = gelu_val * scale_factor
        
        # Update max
        max_val = tl.maximum(max_val, scaled_val)
    
    # Store output
    output_ptrs = output_ptr + offs_m
    tl.store(output_ptrs, max_val, mask=mask_m)


def fused_forward(x, weight, bias, pool_kernel_size, scale_factor):
    M, K = x.shape
    N = weight.shape[0]
    
    output = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    
    fused_matmul_pool_gelu_scale_max_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        pool_kernel_size, scale_factor,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        return fused_forward(x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor)
