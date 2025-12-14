import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l1_norm_kernel(
    x_ptr,
    output_ptr,
    stride_x_batch, stride_x_dim,
    stride_out_batch, stride_out_dim,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Correct and optimized L1 normalization kernel.
    Each block processes one batch element and normalizes along the D dimension.
    """
    pid_batch = tl.program_id(axis=0)
    
    if pid_batch >= B:
        return
    
    # Pointers for the current batch element
    row_ptr = x_ptr + pid_batch * stride_x_batch
    out_row_ptr = output_ptr + pid_batch * stride_out_batch
    
    # Initialize accumulator for absolute sum
    abs_sum = 0.0
    
    # First pass: compute sum of absolute values
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_idx = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_idx < D
        
        x_chunk = tl.load(row_ptr + d_idx * stride_x_dim, mask=mask, other=0.0)
        abs_sum += tl.sum(tl.abs(x_chunk), axis=0)
    
    # Compute mean (avoid division by zero)
    mean_val = abs_sum / D
    # Add small epsilon to prevent division by zero
    mean_val = mean_val + 1e-8
    
    # Second pass: normalize and store
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_idx = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_idx < D
        
        x_chunk = tl.load(row_ptr + d_idx * stride_x_dim, mask=mask, other=0.0)
        normalized = x_chunk / mean_val
        tl.store(out_row_ptr + d_idx * stride_out_dim, normalized, mask=mask)

@triton.autotune(
    configs=[
        # Original config with correct warp sizing
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=32, num_stages=3),
        # Optimized configs based on NCU analysis
        triton.Config({'BLOCK_SIZE_D': 256}, num_warps=8, num_stages=2),  # Reduced stages for memory bound
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=16, num_stages=2),  # Better for memory latency
        triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=32, num_stages=2),  # Maximize occupancy
        triton.Config({'BLOCK_SIZE_D': 512}, num_warps=16, num_stages=4),  # Increased stages for pipelining
        triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=32, num_stages=4),  # Balance memory and compute
    ],
    key=['D'],
)
@triton.jit
def l1_norm_kernel_autotune(
    x_ptr,
    output_ptr,
    stride_x_batch, stride_x_dim,
    stride_out_batch, stride_out_dim,
    B, D,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Autotuned version of the L1 normalization kernel with optimized warp/stage configurations.
    """
    pid_batch = tl.program_id(axis=0)
    
    if pid_batch >= B:
        return
    
    row_ptr = x_ptr + pid_batch * stride_x_batch
    out_row_ptr = output_ptr + pid_batch * stride_out_batch
    
    abs_sum = 0.0
    
    # First pass: compute sum of absolute values
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_idx = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_idx < D
        
        x_chunk = tl.load(row_ptr + d_idx * stride_x_dim, mask=mask, other=0.0)
        abs_sum += tl.sum(tl.abs(x_chunk), axis=0)
    
    # Compute mean with epsilon
    mean_val = abs_sum / D + 1e-8
    
    # Second pass: normalize and store
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_idx = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_idx < D
        
        x_chunk = tl.load(row_ptr + d_idx * stride_x_dim, mask=mask, other=0.0)
        normalized = x_chunk / mean_val
        tl.store(out_row_ptr + d_idx * stride_out_dim, normalized, mask=mask)

def triton_l1_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for L1 normalization with Triton kernels.
    """
    B, D = x.shape
    
    # Ensure contiguous memory layout and proper dtype
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Heuristic for block size selection
    if D <= 1024:  # Adjusted for valid block sizes
        # Use non-autotuned kernel for smaller dimensions
        BLOCK_SIZE_D = min(triton.next_power_of_2(D), 1024)  # Max 1024 threads
        num_warps = BLOCK_SIZE_D // 32  # Correct warp calculation
        
        grid = (B,)
        l1_norm_kernel[grid](
            x_contig, output,
            x_contig.stride(0), x_contig.stride(1),
            output.stride(0), output.stride(1),
            B, D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps,
            num_stages=3,  # Explicitly set default
        )
    else:
        # Use autotuned kernel for larger dimensions
        grid = (B,)
        l1_norm_kernel_autotune[grid](
            x_contig, output,
            x_contig.stride(0), x_contig.stride(1),
            output.stride(0), output.stride(1),
            B, D,
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized L1 normalization layer using high-performance Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x)
