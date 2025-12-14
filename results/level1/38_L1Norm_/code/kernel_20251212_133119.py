import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_fused_kernel_optimized(
    x_ptr,
    output_ptr,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    Optimized fused kernel with improved grid layout.
    Each program processes a 2D tile: BLOCK_SIZE_B rows and BLOCK_SIZE_D columns.
    """
    # 2D program IDs - better parallelism for GPU
    pid_b = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    
    # Calculate tile boundaries
    b_start = pid_b * BLOCK_SIZE_B
    d_start = pid_d * BLOCK_SIZE_D
    
    # Create masks for boundary checking
    b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    
    b_mask = b_offsets < B
    d_mask = d_offsets < D
    
    # Initialize accumulators for row sums (one per row in this tile)
    row_sums = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    
    # First pass: compute sum of absolute values for each row in the tile
    for d_chunk in range(0, D, BLOCK_SIZE_D):
        d_chunk_offsets = d_chunk + tl.arange(0, BLOCK_SIZE_D)
        d_chunk_mask = d_chunk_offsets < D
        
        # Load a chunk of data for all rows in the tile
        # Using 2D loading for better memory access pattern
        x_ptrs = x_ptr + b_offsets[:, None] * stride_x0 + d_chunk_offsets[None, :] * stride_x1
        x_chunk = tl.load(x_ptrs, mask=b_mask[:, None] & d_chunk_mask[None, :], other=0.0)
        
        # Compute absolute values and accumulate row sums
        abs_chunk = tl.abs(x_chunk)
        row_sums += tl.sum(abs_chunk, axis=1)
    
    # Compute mean for each row
    mean_vals = row_sums / D
    
    # Second pass: normalize and store for the assigned column chunk
    # Only process columns assigned to this program
    if pid_d * BLOCK_SIZE_D < D:
        x_ptrs = x_ptr + b_offsets[:, None] * stride_x0 + d_offsets[None, :] * stride_x1
        out_ptrs = output_ptr + b_offsets[:, None] * stride_out0 + d_offsets[None, :] * stride_out1
        
        x_chunk = tl.load(x_ptrs, mask=b_mask[:, None] & d_mask[None, :], other=0.0)
        normalized = x_chunk / mean_vals[:, None]
        tl.store(out_ptrs, normalized, mask=b_mask[:, None] & d_mask[None, :])

@triton.jit
def l1_norm_fused_kernel_large_b(
    x_ptr,
    output_ptr,
    stride_x0, stride_x1,
    stride_out0, stride_out1,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Alternative fused kernel for very large B (better SM utilization).
    Each program processes one row - more parallelism.
    """
    pid_b = tl.program_id(axis=0)
    
    if pid_b >= B:
        return
    
    row_ptr = x_ptr + pid_b * stride_x0
    out_row_ptr = output_ptr + pid_b * stride_out0
    
    # Initialize sum in register
    row_sum = 0.0
    
    # First pass: compute sum of absolute values
    for d_start in range(0, D, BLOCK_SIZE_D):
        offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=mask, other=0.0)
        abs_chunk = tl.abs(x_chunk)
        row_sum += tl.sum(abs_chunk, axis=0)
    
    # Compute mean
    mean_val = row_sum / D
    
    # Second pass: normalize and store
    for d_start in range(0, D, BLOCK_SIZE_D):
        offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask = offsets < D
        
        x_chunk = tl.load(row_ptr + offsets * stride_x1, mask=mask, other=0.0)
        normalized = x_chunk / mean_val
        tl.store(out_row_ptr + offsets * stride_out1, normalized, mask=mask)

def triton_l1_norm_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper function with improved grid strategy.
    """
    B, D = x.shape
    
    # Always use float32 for computation
    if x.dtype != torch.float32:
        x = x.float()
    
    output = torch.empty_like(x)
    
    # Determine optimal kernel based on dimensions
    # For Ada Lovelace with 128 SMs, aim for 2-4x SM count in grid
    
    if B * D <= 1048576:  # Small tensors: use 2D grid for better SM utilization
        BLOCK_SIZE_B = min(32, triton.next_power_of_2(B))
        BLOCK_SIZE_D = min(512, triton.next_power_of_2(D))
        
        grid_b = triton.cdiv(B, BLOCK_SIZE_B)
        grid_d = triton.cdiv(D, BLOCK_SIZE_D)
        
        # Ensure enough blocks for SM utilization (target 256-512 blocks)
        total_blocks = grid_b * grid_d
        if total_blocks < 256:
            # Adjust to get more parallelism
            grid_b = max(grid_b, min(16, triton.cdiv(256, grid_d)))
            total_blocks = grid_b * grid_d
        
        grid = (grid_b, grid_d)
        
        l1_norm_fused_kernel_optimized[grid](
            x, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            B, D,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=8 if D > 8192 else 4
        )
    else:  # Large tensors: use 1D grid for simplicity and good parallelism
        BLOCK_SIZE_D = min(1024, triton.next_power_of_2(D))
        
        # Use 1D grid with one row per block for maximum parallelism
        grid = (B,)
        
        # Adjust number of warps based on D size
        num_warps = 8 if D > 16384 else (4 if D > 4096 else 2)
        
        l1_norm_fused_kernel_large_b[grid](
            x, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            B, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized L1 normalization layer using improved Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        return triton_l1_norm_optimized(x)
