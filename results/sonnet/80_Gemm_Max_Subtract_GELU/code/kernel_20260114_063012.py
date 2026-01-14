import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def gemm_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, partial_max_ptr,
    M, K, N, num_n_blocks,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused GEMM + partial max reduction"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Row indices this program handles
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    
    # Column indices this program handles
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    
    # Accumulator for this block: (BLOCK_M, BLOCK_N)
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
        
        # Matrix multiply
        acc += tl.dot(x_vals, w_vals)
    
    # Add bias
    b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
    acc = acc + b_vals[None, :]
    
    # Mask out invalid columns
    acc = tl.where(n_mask[None, :], acc, -float('inf'))
    
    # Compute max over N dimension for this block: (BLOCK_M,)
    block_max = tl.max(acc, axis=1)
    
    # Store partial max
    partial_offs = pid_m * BLOCK_M * num_n_blocks + m_offs * num_n_blocks + pid_n
    partial_mask = m_mask
    tl.store(partial_max_ptr + partial_offs, block_max, mask=partial_mask)


@triton.jit
def reduce_max_kernel(
    partial_max_ptr, out_ptr,
    M, num_n_blocks,
    BLOCK_M: tl.constexpr,
):
    """Reduce partial maxes to final output"""
    pid = tl.program_id(0)
    
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    
    max_vals = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    
    for n_block in range(num_n_blocks):
        partial_offs = m_offs * num_n_blocks + n_block
        partial_vals = tl.load(partial_max_ptr + partial_offs, mask=m_mask, other=-float('inf'))
        max_vals = tl.maximum(max_vals, partial_vals)
    
    tl.store(out_ptr + m_offs, max_vals, mask=m_mask)


def gemm_max_forward(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    
    # Determine block sizes (will be set by autotune)
    BLOCK_M = 32
    BLOCK_N = 64
    
    num_m_blocks = triton.cdiv(M, BLOCK_M)
    num_n_blocks = triton.cdiv(N, BLOCK_N)
    
    # Allocate partial max buffer
    partial_max = torch.empty((M, num_n_blocks), device=x.device, dtype=torch.float32)
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    # Launch GEMM + partial max kernel
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    gemm_max_kernel[grid](
        x, weight, bias, out, partial_max,
        M, K, N, num_n_blocks,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    
    # Get actual num_n_blocks from autotuned config
    actual_num_n_blocks = triton.cdiv(N, 64)  # Use conservative estimate
    
    # Launch reduction kernel
    REDUCE_BLOCK_M = 64
    reduce_grid = (triton.cdiv(M, REDUCE_BLOCK_M),)
    
    reduce_max_kernel[reduce_grid](
        partial_max, out,
        M, actual_num_n_blocks,
        BLOCK_M=REDUCE_BLOCK_M,
    )
    
    return out.unsqueeze(1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def gemm_max_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fully fused GEMM + max reduction - single kernel"""
    pid_m = tl.program_id(0)
    
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    
    max_vals = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            
            x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_mask = m_mask[:, None] & k_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
            
            w_ptrs = w_ptr + k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            w_mask = k_mask[:, None] & n_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            acc += tl.dot(x_vals, w_vals)
        
        b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        acc = acc + b_vals[None, :]
        acc = tl.where(n_mask[None, :], acc, -float('inf'))
        
        block_max = tl.max(acc, axis=1)
        max_vals = tl.maximum(max_vals, block_max)
    
    tl.store(out_ptr + m_offs, max_vals, mask=m_mask)


def gemm_max_forward_fused(x, weight, bias):
    M, K = x.shape
    N = weight.shape[1]
    
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)
    
    gemm_max_fused_kernel[grid](
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
        weight = self.gemm.weight.t().contiguous()
        bias = self.gemm.bias
        
        if self.max_dim == 1:
            return gemm_max_forward_fused(x, weight, bias)
        else:
            x = self.gemm(x)
            x = torch.max(x, dim=self.max_dim)[0]
            return x.unsqueeze(self.max_dim)
