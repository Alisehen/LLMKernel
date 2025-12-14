import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 4}),
        triton.Config({'num_warps': 16, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
    ],
    key=['n_rows', 'n_cols', 'BLOCK_K'],
)
@triton.jit
def log_softmax_kernel_large_optimized(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_x,
    stride_output,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """LogSoftmax kernel optimized for large columns with fused passes."""
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Row pointers
    x_row_ptr = x_ptr + row_idx * stride_x
    output_row_ptr = output_ptr + row_idx * stride_output
    
    # Fused pass: compute max and sum_exp simultaneously
    row_max = -float('inf')
    row_sum_exp = 0.0
    
    # Precompute offsets for better memory coalescing
    col_offsets_base = tl.arange(0, BLOCK_K)
    
    # First pass: compute max only (better cache locality)
    for col_start in range(0, n_cols, BLOCK_K):
        col_offsets = col_start + col_offsets_base
        col_mask = col_offsets < n_cols
        
        # Single load per iteration
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=col_mask, other=-float('inf'))
        
        # Compute local max
        local_max = tl.max(x_chunk, axis=0)
        row_max = tl.maximum(row_max, local_max)
    
    # Second pass: compute sum_exp with better numerical stability
    for col_start in range(0, n_cols, BLOCK_K):
        col_offsets = col_start + col_offsets_base
        col_mask = col_offsets < n_cols
        
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=col_mask, other=0.0)
        
        # Compute exp(x - max) with numerical stability
        exp_val = tl.exp(x_chunk - row_max)
        row_sum_exp += tl.sum(exp_val, axis=0)
    
    # Compute logsumexp
    log_sum_exp = tl.log(row_sum_exp) + row_max
    
    # Third pass: compute and store output
    for col_start in range(0, n_cols, BLOCK_K):
        col_offsets = col_start + col_offsets_base
        col_mask = col_offsets < n_cols
        
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=col_mask)
        output = x_chunk - log_sum_exp
        tl.store(output_row_ptr + col_offsets, output, mask=col_mask)

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2, 'num_stages': 2}),
        triton.Config({'num_warps': 2, 'num_stages': 3}),
        triton.Config({'num_warps': 4, 'num_stages': 2}),
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
    ],
    key=['n_rows', 'n_cols', 'BLOCK_K', 'BLOCK_ROWS'],
)
@triton.jit
def log_softmax_kernel_small_optimized(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_x0,
    stride_x1,
    stride_output0,
    stride_output1,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    num_stages: tl.constexpr,
    num_warps: tl.constexpr,
):
    """LogSoftmax kernel for small columns with improved memory access patterns."""
    row_block_idx = tl.program_id(0)
    row_start = row_block_idx * BLOCK_ROWS
    
    # Precompute column offsets and mask
    col_offsets = tl.arange(0, BLOCK_K)
    col_mask = col_offsets < n_cols
    
    # Allocate shared memory for intermediate computations
    shmem_max = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    shmem_sum = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    
    for r in range(BLOCK_ROWS):
        row_idx = row_start + r
        if row_idx < n_rows:
            x_row_ptr = x_ptr + row_idx * stride_x0
            x_row = tl.load(x_row_ptr + col_offsets * stride_x1, mask=col_mask)
            
            # Compute row max
            row_max = tl.max(x_row, axis=0)
            shmem_max = tl.where(tl.arange(0, BLOCK_ROWS) == r, row_max, shmem_max)
            
            # Compute row sum_exp
            exp_vals = tl.exp(x_row - row_max)
            row_sum_exp = tl.sum(exp_vals, axis=0)
            shmem_sum = tl.where(tl.arange(0, BLOCK_ROWS) == r, row_sum_exp, shmem_sum)
    
    # Compute logsumexp and store results
    for r in range(BLOCK_ROWS):
        row_idx = row_start + r
        if row_idx < n_rows:
            x_row_ptr = x_ptr + row_idx * stride_x0
            output_row_ptr = output_ptr + row_idx * stride_output0
            
            x_row = tl.load(x_row_ptr + col_offsets * stride_x1, mask=col_mask)
            
            row_max = shmem_max[r]
            row_sum_exp = shmem_sum[r]
            log_sum_exp = tl.log(row_sum_exp) + row_max
            
            output = x_row - log_sum_exp
            tl.store(output_row_ptr + col_offsets * stride_output1, output, mask=col_mask)

def triton_log_softmax_optimized(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Optimized Triton-accelerated LogSoftmax function."""
    # Ensure tensor is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    original_shape = x.shape
    output = torch.empty_like(x)
    
    # Handle different dimensions
    if x.dim() == 2:
        n_rows, n_cols = x.shape
    else:
        if dim < 0:
            dim = x.dim() + dim
        
        n_rows = 1
        for i in range(dim):
            n_rows *= x.shape[i]
        
        n_cols = 1
        for i in range(dim, x.dim()):
            n_cols *= x.shape[i]
        
        x = x.view(n_rows, n_cols)
        output = output.view(n_rows, n_cols)
    
    MAX_BLOCK_K = 2048
    
    # Choose kernel based on column size
    if n_cols <= MAX_BLOCK_K:
        # Use optimized small kernel with better parameters
        BLOCK_K = min(MAX_BLOCK_K, triton.next_power_of_2(n_cols))
        
        # Dynamically determine BLOCK_ROWS based on column size
        if BLOCK_K <= 256:
            BLOCK_ROWS = 16
        elif BLOCK_K <= 512:
            BLOCK_ROWS = 8
        elif BLOCK_K <= 1024:
            BLOCK_ROWS = 4
        else:
            BLOCK_ROWS = 2
        
        # Ensure we don't exceed thread block limits
        max_threads = BLOCK_K * BLOCK_ROWS
        if max_threads > 1024:
            BLOCK_ROWS = max(1, 1024 // BLOCK_K)
        
        grid_rows = (n_rows + BLOCK_ROWS - 1) // BLOCK_ROWS
        
        # Launch optimized small kernel with autotune
        log_softmax_kernel_small_optimized[(grid_rows,)](
            x, output,
            n_rows, n_cols,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_K=BLOCK_K,
            BLOCK_ROWS=BLOCK_ROWS,
        )
    else:
        # Use optimized large kernel with improved parameters
        BLOCK_K = max(256, min(MAX_BLOCK_K, triton.next_power_of_2(min(n_cols, 2048))))
        
        # Grid size is number of rows
        grid = (n_rows,)
        
        # Launch optimized large kernel with autotune
        log_softmax_kernel_large_optimized[grid](
            x, output,
            n_rows, n_cols,
            x.stride(0),
            output.stride(0),
            BLOCK_K=BLOCK_K,
        )
    
    # Restore original shape if needed
    if output.shape != original_shape:
        output = output.view(original_shape)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs LogSoftmax activation using Triton kernels.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation using optimized Triton kernels.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        return triton_log_softmax_optimized(x, dim=self.dim)
