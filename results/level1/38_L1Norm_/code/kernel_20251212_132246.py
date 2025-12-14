import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_reduce_kernel(
    x_ptr,
    sum_ptr,
    stride_x0, stride_x1,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    First pass: compute sum of absolute values for each row.
    Each program processes one row (B parallel).
    """
    pid_b = tl.program_id(axis=0)
    
    if pid_b >= B:
        return
    
    row_ptr = x_ptr + pid_b * stride_x0
    sum_row = tl.zeros([1], dtype=tl.float32)
    
    # Process row in chunks
    for d_start in range(0, D, BLOCK_SIZE_D):
        offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=mask, other=0.0)
        abs_chunk = tl.abs(x_chunk)
        sum_row += tl.sum(abs_chunk, axis=0)
    
    # Store row sum
    tl.store(sum_ptr + pid_b, sum_row)

@triton.jit
def l1_norm_divide_kernel(
    x_ptr,
    sum_ptr,
    output_ptr,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Second pass: divide each element by (sum / D) for normalization.
    Each program processes a column chunk for all rows.
    """
    pid_d = tl.program_id(axis=0)
    
    d_start = pid_d * BLOCK_SIZE_D
    offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    col_mask = offsets < D
    
    # Process all rows for this column chunk
    for b in range(B):
        row_ptr = x_ptr + b * stride_x0
        out_row_ptr = output_ptr + b * stride_out0
        
        # Load row sum and compute mean
        row_sum = tl.load(sum_ptr + b)
        mean_val = row_sum / D
        
        # Load data, normalize, and store
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=col_mask, other=0.0)
        normalized = tl.where(col_mask, x_chunk / mean_val, 0.0)
        tl.store(out_row_ptr + offsets * stride_out1, normalized, mask=col_mask)

@triton.jit
def l1_norm_fused_kernel(
    x_ptr,
    output_ptr,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
):
    """
    Fused kernel that computes L1 normalization in one pass using shared memory.
    For best performance with moderate B and large D.
    """
    pid_b = tl.program_id(axis=0)
    
    if pid_b >= B:
        return
    
    row_ptr = x_ptr + pid_b * stride_x0
    out_row_ptr = output_ptr + pid_b * stride_out0
    
    # Initialize sum in register
    row_sum = 0.0
    
    # First: compute sum of absolute values
    for d_start in range(0, D, BLOCK_SIZE_D):
        offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=mask, other=0.0)
        abs_chunk = tl.abs(x_chunk)
        row_sum += tl.sum(abs_chunk, axis=0)
    
    # Compute mean
    mean_val = row_sum / D
    
    # Second: normalize and store
    for d_start in range(0, D, BLOCK_SIZE_D):
        offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=mask, other=0.0)
        normalized = tl.where(mask, x_chunk / mean_val, 0.0)
        tl.store(out_row_ptr + offsets * stride_out1, normalized, mask=mask)

def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for L1 normalization using Triton kernels.
    """
    B, D = x.shape
    
    # Always use float32 for computation to avoid precision issues
    if x.dtype != torch.float32:
        x = x.float()
    
    output = torch.empty_like(x)
    
    # Choose optimal kernel based on dimensions
    # For small to moderate B (<= 8192), use fused kernel for best performance
    # For large B, use two-pass approach to reduce register pressure
    if B <= 8192:
        BLOCK_SIZE_D = 1024
        grid = (B,)
        l1_norm_fused_kernel[grid](
            x, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            B, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            USE_ATOMICS=False,
            num_warps=8 if D > 4096 else 4
        )
    else:
        # Two-pass approach for large B
        BLOCK_SIZE_D = 1024
        
        # First pass: compute row sums
        row_sums = torch.empty(B, device=x.device, dtype=torch.float32)
        grid_reduce = (triton.cdiv(B, 1),)
        l1_norm_reduce_kernel[grid_reduce](
            x, row_sums,
            x.stride(0), x.stride(1),
            B, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=4
        )
        
        # Second pass: normalize
        grid_divide = (triton.cdiv(D, BLOCK_SIZE_D),)
        l1_norm_divide_kernel[grid_divide](
            x, row_sums, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            B, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=4
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized L1 normalization layer using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        return triton_l1_norm(x)
