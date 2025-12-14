import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def log_softmax_kernel(
    x_ptr,
    output_ptr,
    n_cols,
    stride_x,
    stride_output,
    BLOCK_SIZE: tl.constexpr,
):
    """LogSoftmax kernel optimized for row-wise reduction."""
    # Row index
    row_idx = tl.program_id(axis=0)
    
    # Compute base pointers for this row
    x_row_ptr = x_ptr + row_idx * stride_x
    output_row_ptr = output_ptr + row_idx * stride_output
    
    # Step 1: Find row maximum for numerical stability
    col_start = 0
    max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
    
    # Process in blocks to handle large columns
    while col_start < n_cols:
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        chunk_max = tl.max(x_chunk, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
        
        col_start += BLOCK_SIZE
    
    # Step 2: Compute sum of exp(x - max)
    col_start = 0
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    
    while col_start < n_cols:
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=mask)
        exp_val = tl.exp(x_chunk - max_val)
        sum_exp += tl.sum(exp_val, axis=0)
        
        col_start += BLOCK_SIZE
    
    # Step 3: Compute log(sum(exp(x - max))) = logsumexp
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Step 4: Compute log_softmax = x - logsumexp
    col_start = 0
    while col_start < n_cols:
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_chunk = tl.load(x_row_ptr + col_offsets, mask=mask)
        output = x_chunk - log_sum_exp
        tl.store(output_row_ptr + col_offsets, output, mask=mask)
        
        col_start += BLOCK_SIZE

@triton.jit
def log_softmax_kernel_small(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_x0,
    stride_x1,
    stride_output0,
    stride_output1,
    BLOCK_SIZE: tl.constexpr,
):
    """LogSoftmax kernel for when all columns fit in one block."""
    # Row index
    row_idx = tl.program_id(axis=0)
    
    # Check if row is valid
    if row_idx >= n_rows:
        return
    
    # Load entire row at once
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Compute base pointers for this row
    x_row_ptr = x_ptr + row_idx * stride_x0
    output_row_ptr = output_ptr + row_idx * stride_output0
    
    # Load entire row
    x_row = tl.load(x_row_ptr + col_offsets * stride_x1, mask=mask, other=-float('inf'))
    
    # Step 1: Find row maximum for numerical stability
    row_max = tl.max(x_row, axis=0)
    
    # Step 2: Compute sum of exp(x - max)
    exp_vals = tl.exp(x_row - row_max)
    row_sum_exp = tl.sum(exp_vals, axis=0)
    
    # Step 3: Compute log_softmax
    log_sum_exp = tl.log(row_sum_exp) + row_max
    output = x_row - log_sum_exp
    
    # Store results
    tl.store(output_row_ptr + col_offsets * stride_output1, output, mask=mask)

def triton_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Triton-accelerated LogSoftmax function."""
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert dim == 1 or dim == -1, "Only dim=1 or dim=-1 is supported"
    
    # Save original shape
    original_shape = x.shape
    output = torch.empty_like(x)
    
    # Get tensor dimensions
    if x.dim() == 2:
        n_rows, n_cols = x.shape
    else:
        # Handle higher dimensions by flattening
        if dim < 0:
            dim = x.dim() + dim
        
        # Flatten all dimensions before dim into n_rows
        n_rows = 1
        for i in range(dim):
            n_rows *= x.shape[i]
        
        # Flatten all dimensions from dim onward into n_cols
        n_cols = 1
        for i in range(dim, x.dim()):
            n_cols *= x.shape[i]
        
        x = x.view(n_rows, n_cols)
        output = output.view(n_rows, n_cols)
    
    # Choose optimal block size based on column size
    MAX_BLOCK_SIZE = 4096
    BLOCK_SIZE = min(MAX_BLOCK_SIZE, triton.next_power_of_2(n_cols))
    
    # Choose kernel based on column size
    if n_cols <= MAX_BLOCK_SIZE:
        # Use kernel that processes entire row in one block
        grid = (n_rows,)
        log_softmax_kernel_small[grid](
            x, output,
            n_rows, n_cols,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Use kernel that processes row in blocks
        grid = (n_rows,)
        log_softmax_kernel[grid](
            x, output,
            n_cols,
            x.stride(0),
            output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
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
        Applies LogSoftmax activation using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied.
        """
        return triton_log_softmax(x, dim=self.dim)
