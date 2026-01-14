import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_sum_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    # Use scalar accumulator
    total_sum = 0.0
    
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        mask_n = n_offs < N
        
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < K
            
            # Load x values for this row
            x_vals = tl.load(x_ptr + pid_m * stride_xm + k_offs * stride_xk, mask=mask_k, other=0.0)
            
            # Load weight block [BLOCK_N, BLOCK_K]
            w_ptrs = weight_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            
            # Compute partial dot products
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        
        # Add bias
        bias_vals = tl.load(bias_ptr + n_offs, mask=mask_n, other=0.0)
        acc += bias_vals
        
        # Sum up this block's contribution
        total_sum += tl.sum(tl.where(mask_n, acc, 0.0))
    
    # Store result - use offset array for proper store
    out_offs = tl.arange(0, 1)
    tl.store(out_ptr + pid_m + out_offs, tl.full((1,), total_sum, dtype=tl.float32))


@triton.jit
def fused_linear_sum_kernel_simple(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row of the batch
    pid_m = tl.program_id(0)
    
    # Scalar accumulator for the final sum
    total_sum = 0.0
    
    offs_k = tl.arange(0, BLOCK_K)
    
    # Process all output features one by one
    for n in range(N):
        acc = 0.0
        # Compute dot product for this output feature
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            mask_k = k_offs < K
            
            x_vals = tl.load(x_ptr + pid_m * stride_xm + k_offs * stride_xk, mask=mask_k, other=0.0)
            w_vals = tl.load(weight_ptr + n * stride_wn + k_offs * stride_wk, mask=mask_k, other=0.0)
            
            acc += tl.sum(x_vals * w_vals)
        
        # Add bias
        bias_val = tl.load(bias_ptr + n)
        acc += bias_val
        total_sum += acc
    
    # Store result using proper block store
    out_offs = tl.arange(0, 1)
    tl.store(out_ptr + pid_m + out_offs, tl.full((1,), total_sum, dtype=tl.float32))


def fused_linear_sum(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    
    BLOCK_K = 128
    
    grid = (M,)
    fused_linear_sum_kernel_simple[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_K=BLOCK_K,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Fused: linear + sum (subsequent ops are identity on scalar)
        return fused_linear_sum(x, self.linear.weight, self.linear.bias)
