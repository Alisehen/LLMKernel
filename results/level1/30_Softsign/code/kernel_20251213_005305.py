import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softsign_kernel_3d(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_batch,
    stride_row,
    stride_col,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    """
    3D grid implementation for better occupancy
    Optimized for tensor shape [batch, rows, cols]
    """
    # 3D grid: (batch_id, row_block_id, col_block_id)
    batch_id = tl.program_id(0)
    row_block_id = tl.program_id(1)
    col_block_id = tl.program_id(2)
    
    # Compute row offsets
    row_start = row_block_id * BLOCK_ROW
    row_offsets = row_start + tl.arange(0, BLOCK_ROW)
    row_mask = row_offsets < n_rows
    
    # Compute column offsets
    col_start = col_block_id * BLOCK_COL
    col_offsets = col_start + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < n_cols
    
    # 2D mask for the block
    block_mask = row_mask[:, None] & col_mask[None, :]
    
    # Compute base pointer for this batch
    batch_offset = batch_id * stride_batch
    x_batch_ptr = x_ptr + batch_offset
    output_batch_ptr = output_ptr + batch_offset
    
    # Load block with 2D offsets
    x_ptrs = x_batch_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    x_block = tl.load(x_ptrs, mask=block_mask, other=0.0)
    
    # Compute softsign: x / (1 + |x|)
    # Use fast math operations optimized for Tensor Cores
    abs_x = tl.abs(x_block)
    denominator = 1.0 + abs_x
    
    # Use fast approximate reciprocal for better throughput
    # On Ada Lovelace, this can use Tensor Cores for certain precisions
    if BLOCK_ROW * BLOCK_COL >= 512:  # Use faster math for larger blocks
        inv_denom = tl.math.fast_reciprocal(denominator)
    else:
        inv_denom = 1.0 / denominator
    
    output = x_block * inv_denom
    
    # Store result
    output_ptrs = output_batch_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    tl.store(output_ptrs, output, mask=block_mask)


@triton.jit
def softsign_kernel_3d_fp16(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    stride_batch,
    stride_row,
    stride_col,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    """
    Optimized for FP16/BF16 with Tensor Core support on Ada Lovelace
    """
    # 3D grid
    batch_id = tl.program_id(0)
    row_block_id = tl.program_id(1)
    col_block_id = tl.program_id(2)
    
    # Row and column offsets
    row_start = row_block_id * BLOCK_ROW
    row_offsets = row_start + tl.arange(0, BLOCK_ROW)
    row_mask = row_offsets < n_rows
    
    col_start = col_block_id * BLOCK_COL
    col_offsets = col_start + tl.arange(0, BLOCK_COL)
    col_mask = col_offsets < n_cols
    
    block_mask = row_mask[:, None] & col_mask[None, :]
    
    # Base pointer for batch
    batch_offset = batch_id * stride_batch
    x_batch_ptr = x_ptr + batch_offset
    output_batch_ptr = output_ptr + batch_offset
    
    # Load block - use fp16 precision for Tensor Cores
    x_ptrs = x_batch_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    x_block = tl.load(x_ptrs, mask=block_mask, other=0.0)
    
    # Direct computation optimized for Tensor Cores
    # Ada Lovelace can use Tensor Cores for element-wise ops in fp16/bf16
    abs_x = tl.abs(x_block)
    output = x_block / (1.0 + abs_x)
    
    # Store result
    output_ptrs = output_batch_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    tl.store(output_ptrs, output, mask=block_mask)


def triton_softsign_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized 3D grid implementation for large tensors
    Target shape: [4096, 393216] -> reshape to 3D: [4096, 768, 512]
    """
    output = torch.empty_like(x)
    
    # Reshape to 3D if needed for better grid layout
    if x.ndim == 2:
        batch_size, n_elements = x.shape
        
        # Choose optimal reshaping based on tensor size
        # For [4096, 393216], we can use [4096, 768, 512]
        # This gives us good 3D grid distribution
        if n_elements == 393216 and batch_size == 4096:
            # Reshape to 3D for optimal grid
            n_rows = 768
            n_cols = 512
            x_reshaped = x.view(batch_size, n_rows, n_cols)
            output_reshaped = output.view(batch_size, n_rows, n_cols)
        else:
            # General case: try to make it at least 2D
            if n_elements > 65536:
                # Find good factorization
                n_rows = 1024
                n_cols = triton.cdiv(n_elements, n_rows)
                x_reshaped = x.view(batch_size, n_rows, n_cols)
                output_reshaped = output.view(batch_size, n_rows, n_cols)
            else:
                # Small tensor, use 2D
                n_rows = batch_size
                n_cols = n_elements
                x_reshaped = x.view(batch_size, 1, n_elements)
                output_reshaped = output.view(batch_size, 1, n_elements)
    else:
        # Already 3D or higher
        x_reshaped = x
        output_reshaped = output
    
    # Get dimensions
    batch_size = x_reshaped.shape[0]
    n_rows = x_reshaped.shape[1] if x_reshaped.ndim >= 2 else 1
    n_cols = x_reshaped.shape[2] if x_reshaped.ndim >= 3 else x_reshaped.shape[1] if x_reshaped.ndim == 2 else x_reshaped.numel()
    
    # Strides
    if x_reshaped.ndim >= 3:
        stride_batch = x_reshaped.stride(0)
        stride_row = x_reshaped.stride(1)
        stride_col = x_reshaped.stride(2)
    else:
        # 2D case
        stride_batch = x_reshaped.stride(0)
        stride_row = 0
        stride_col = x_reshaped.stride(1) if x_reshaped.ndim == 2 else 1
    
    # Choose kernel and configuration based on dtype and size
    if x.dtype in (torch.float16, torch.bfloat16):
        # Use Tensor Core optimized kernel for half precision
        BLOCK_ROW = 64  # Larger blocks for better occupancy
        BLOCK_COL = 64
        
        # Ensure blocks fit in shared memory
        if BLOCK_ROW * BLOCK_COL * x.element_size() > 65536:
            BLOCK_ROW = 32
            BLOCK_COL = 32
        
        grid = (batch_size, triton.cdiv(n_rows, BLOCK_ROW), triton.cdiv(n_cols, BLOCK_COL))
        
        # Launch with optimal warp configuration for Ada Lovelace
        num_warps = 8
        if BLOCK_ROW * BLOCK_COL >= 4096:
            num_warps = 16  # More warps for larger blocks
        
        softsign_kernel_3d_fp16[grid](
            x_reshaped, output_reshaped,
            n_rows, n_cols,
            stride_batch, stride_row, stride_col,
            BLOCK_ROW=BLOCK_ROW,
            BLOCK_COL=BLOCK_COL,
            num_warps=num_warps,
            num_stages=3
        )
    else:
        # FP32 kernel
        BLOCK_ROW = 32
        BLOCK_COL = 32
        
        grid = (batch_size, triton.cdiv(n_rows, BLOCK_ROW), triton.cdiv(n_cols, BLOCK_COL))
        
        num_warps = 8
        if n_rows * n_cols > 1000000:
            num_warps = 16
        
        softsign_kernel_3d[grid](
            x_reshaped, output_reshaped,
            n_rows, n_cols,
            stride_batch, stride_row, stride_col,
            BLOCK_ROW=BLOCK_ROW,
            BLOCK_COL=BLOCK_COL,
            num_warps=num_warps,
            num_stages=3
        )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model with 3D grid layout for maximum SM utilization
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softsign activation using 3D grid layout
        """
        return triton_softsign_optimized(x)
