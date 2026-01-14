import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused GEMM + max reduction along dim=1 (output features)"""
    pid_m = tl.program_id(0)
    
    # Each program handles one row of output
    row_idx = pid_m
    
    if row_idx >= M:
        return
    
    # Initialize max value to very negative
    max_val = -float('inf')
    
    # Iterate over output columns in blocks
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        
        # Compute dot product for this block of output columns
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        
        # Iterate over K dimension
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            
            # Load x[row_idx, k_offs]
            x_ptrs = x_ptr + row_idx * stride_xm + k_offs * stride_xk
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0)
            
            # Load w[k_offs, n_offs] - shape (BLOCK_K, BLOCK_N)
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_mask = k_mask[:, None] & n_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            # Accumulate: (BLOCK_K,) @ (BLOCK_K, BLOCK_N) -> (BLOCK_N,)
            acc += tl.sum(x_vals[:, None] * w_vals, axis=0)
        
        # Add bias
        b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        acc = acc + b_vals
        
        # Update max - mask out invalid positions
        acc = tl.where(n_mask, acc, -float('inf'))
        block_max = tl.max(acc, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    tl.store(out_ptr + row_idx, max_val)


def gemm_max_forward(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 1
    BLOCK_K = 64
    BLOCK_N = 64
    
    grid = (M,)
    
    gemm_max_kernel[grid](
        x, weight, bias, out,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    
    return out.unsqueeze(1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        # Get weight in correct layout (K, N) where K=in_features, N=out_features
        # nn.Linear stores weight as (out_features, in_features), so we transpose
        weight = self.gemm.weight.t().contiguous()
        bias = self.gemm.bias
        
        # Fused GEMM + max reduction
        # Assuming max_dim=1 (reduce over output features)
        if self.max_dim == 1:
            return gemm_max_forward(x, weight, bias)
        else:
            # Fallback for other dimensions
            x = self.gemm(x)
            x = torch.max(x, dim=self.max_dim)[0]
            return x.unsqueeze(self.max_dim)
