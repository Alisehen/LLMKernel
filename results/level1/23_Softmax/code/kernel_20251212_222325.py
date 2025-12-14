import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_max_kernel(
    x_ptr,
    max_ptr,
    stride_x0,
    stride_x1,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute max along dim=1 for 2D tensor."""
    # Each program processes a row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Row offset in x
    row_start_ptr = x_ptr + row_idx * stride_x0
    
    # Initialize max
    row_max = -tl.inf
    
    # Process columns in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_indices = block_start + col_offsets
        mask = col_indices < n_cols
        
        # Load block from current row
        ptrs = row_start_ptr + col_indices * stride_x1
        x_block = tl.load(ptrs, mask=mask, other=-tl.inf)
        
        # Update max
        row_max = tl.maximum(row_max, tl.max(x_block, 0))
    
    # Store row max
    tl.store(max_ptr + row_idx, row_max)

@triton.jit
def softmax_exp_sum_kernel(
    x_ptr,
    max_ptr,
    output_ptr,
    sum_ptr,
    stride_x0,
    stride_x1,
    stride_out0,
    stride_out1,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute exp(x - max) and sum along dim=1 for 2D tensor."""
    # Each program processes a row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Get row max
    row_max = tl.load(max_ptr + row_idx)
    
    # Row offsets
    row_start_x = x_ptr + row_idx * stride_x0
    row_start_out = output_ptr + row_idx * stride_out0
    
    # Initialize sum
    row_sum = 0.0
    
    # Process columns in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_indices = block_start + col_offsets
        mask = col_indices < n_cols
        
        # Load block from current row
        ptrs = row_start_x + col_indices * stride_x1
        x_block = tl.load(ptrs, mask=mask, other=0.0)
        
        # Compute exp(x - max)
        shifted = x_block - row_max
        exp_block = tl.exp(shifted)
        
        # Update sum
        row_sum += tl.sum(exp_block, 0)
        
        # Store exp values
        out_ptrs = row_start_out + col_indices * stride_out1
        tl.store(out_ptrs, exp_block, mask=mask)
    
    # Store row sum
    tl.store(sum_ptr + row_idx, row_sum)

@triton.jit
def softmax_normalize_kernel(
    exp_ptr,
    sum_ptr,
    output_ptr,
    stride_exp0,
    stride_exp1,
    stride_out0,
    stride_out1,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize exp values by sum along dim=1."""
    # Each program processes a row
    row_idx = tl.program_id(0)
    
    if row_idx >= n_rows:
        return
    
    # Get row sum
    row_sum = tl.load(sum_ptr + row_idx)
    
    # Avoid division by zero
    inv_sum = 1.0 / (row_sum + 1e-8)
    
    # Row offsets
    row_start_exp = exp_ptr + row_idx * stride_exp0
    row_start_out = output_ptr + row_idx * stride_out0
    
    # Process columns in blocks
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_indices = block_start + col_offsets
        mask = col_indices < n_cols
        
        # Load exp block
        exp_ptrs = row_start_exp + col_indices * stride_exp1
        exp_block = tl.load(exp_ptrs, mask=mask, other=0.0)
        
        # Normalize
        output_block = exp_block * inv_sum
        
        # Store normalized values
        out_ptrs = row_start_out + col_indices * stride_out1
        tl.store(out_ptrs, output_block, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax along dim=1 using Triton kernels.
    Strategy: max -> exp -> sum -> normalize
    """
    # Allocate output tensor
    output = torch.empty_like(x)
    batch_size, n_cols = x.shape
    
    # Allocate intermediate buffers
    max_vals = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    row_sums = torch.empty(batch_size, dtype=x.dtype, device=x.device)
    
    # Allocate temporary exp tensor
    exp_vals = torch.empty_like(x)
    
    # Choose optimal block size
    # For large columns, use large block size for better memory coalescing
    BLOCK_SIZE = 1024 if n_cols % 1024 == 0 else 512
    
    # Launch kernels
    # 1. Compute max per row
    grid = lambda meta: (batch_size,)
    softmax_max_kernel[grid](
        x, max_vals,
        x.stride(0), x.stride(1),
        batch_size, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 2. Compute exp(x - max) and sum per row
    softmax_exp_sum_kernel[grid](
        x, max_vals, exp_vals, row_sums,
        x.stride(0), x.stride(1),
        exp_vals.stride(0), exp_vals.stride(1),
        batch_size, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 3. Normalize by sum
    softmax_normalize_kernel[grid](
        exp_vals, row_sums, output,
        exp_vals.stride(0), exp_vals.stride(1),
        output.stride(0), output.stride(1),
        batch_size, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation using optimized Triton kernels.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor along dim=1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return triton_softmax(x)
