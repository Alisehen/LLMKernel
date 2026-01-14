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
    
    N_pooled = N // pool_kernel_size
    
    # Allocate shared memory for pooled values
    pooled_vals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Process in blocks of BLOCK_N pooled outputs at a time
    for pool_block_start in range(0, N_pooled, BLOCK_N):
        pool_block_end = tl.minimum(pool_block_start + BLOCK_N, N_pooled)
        pool_block_size = pool_block_end - pool_block_start
        
        # Initialize accumulator for this pool block
        pool_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # For each pooled output in this block
        for pool_idx in range(BLOCK_N):
            actual_pool_idx = pool_block_start + pool_idx
            
            # Skip if beyond N_pooled
            if actual_pool_idx >= N_pooled:
                continue
            
            # Accumulate over pool_kernel_size elements
            for k_pool in range(pool_kernel_size):
                n_idx = actual_pool_idx * pool_kernel_size + k_pool
                
                # Compute matmul for this specific output column
                offs_k = tl.arange(0, BLOCK_K)
                x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
                w_ptrs = weight_ptr + n_idx * stride_wn + offs_k * stride_wk
                
                acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
                for k in range(0, K, BLOCK_K):
                    mask_k = offs_k < K - k
                    x_block = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
                    w_block = tl.load(w_ptrs, mask=mask_k, other=0.0)
                    acc += tl.sum(x_block * w_block[None, :], axis=1)
                    x_ptrs += BLOCK_K * stride_xk
                    w_ptrs += BLOCK_K * stride_wk
                
                # Add bias
                bias_val = tl.load(bias_ptr + n_idx)
                acc += bias_val
                
                # Accumulate into pool
                pool_acc[:, pool_idx] += acc
        
        # Apply average pooling
        pool_avg = pool_acc / tl.cast(pool_kernel_size, tl.float32)
        
        # Apply GELU
        sqrt_2_over_pi = 0.7978845608028654
        x = pool_avg
        x3 = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * x3)
        
        inner = tl.where(inner > 10.0, 10.0, inner)
        inner = tl.where(inner < -10.0, -10.0, inner)
        
        exp_2x = tl.exp(2.0 * inner)
        tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
        gelu_val = 0.5 * x * (1.0 + tanh_inner)
        
        # Apply scaling
        scaled_val = gelu_val * scale_factor
        
        # Store in pooled_vals for later max reduction
        pooled_vals[:, pool_block_start:pool_block_end] = scaled_val[:, :pool_block_size]
    
    # Max reduction across all pooled values
    max_val = tl.max(pooled_vals, axis=1)
    
    # Store output
    output_ptrs = output_ptr + offs_m
    tl.store(output_ptrs, max_val, mask=offs_m < M)


def fused_forward(x, weight, bias, pool_kernel_size, scale_factor):
    M, K = x.shape
    N = weight.shape[0]
    
    output = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    N_pooled = N // pool_kernel_size
    BLOCK_M = 64
    BLOCK_N = min(64, triton.next_power_of_2(N_pooled))
    BLOCK_K = 32
    
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
