import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'K', 'N'],
)
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
    
    # Row indices this program handles
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    
    # Initialize max values for each row
    max_vals = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    
    # Iterate over output columns in blocks
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        
        # Accumulator for this block of output columns: (BLOCK_M, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Iterate over K dimension
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            
            # Load x[m_offs, k_offs] - shape (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
            
            # Load w[k_offs, n_offs] - shape (BLOCK_K, BLOCK_N)
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_mask = k_mask[:, None] & n_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            # Matrix multiply: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
            acc += tl.dot(x_vals, w_vals)
        
        # Add bias: broadcast (BLOCK_N,) to (BLOCK_M, BLOCK_N)
        b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        acc = acc + b_vals[None, :]
        
        # Mask out invalid columns for max computation
        acc = tl.where(n_mask[None, :], acc, -float('inf'))
        
        # Compute max over N dimension for this block: (BLOCK_M,)
        block_max = tl.max(acc, axis=1)
        
        # Update running max
        max_vals = tl.maximum(max_vals, block_max)
    
    # Store results
    tl.store(out_ptr + m_offs, max_vals, mask=m_mask)


def gemm_max_forward(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    # Grid: one program per BLOCK_M rows
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)
    
    gemm_max_kernel[grid](
        x, weight, bias, out,
        M, K, N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
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
