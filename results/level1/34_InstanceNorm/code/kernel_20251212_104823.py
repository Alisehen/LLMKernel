import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_moments_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    spatial_size,
    stride_nc,
    eps: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    """Compute mean and variance for multiple (n,c) slices in parallel"""
    pid_spatial = tl.program_id(axis=0)
    pid_nc = tl.program_id(axis=1) * BLOCK_NC
    
    # Create ranges
    nc_offsets = pid_nc + tl.arange(0, BLOCK_NC)
    spatial_offsets = pid_spatial * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    
    # Masks
    nc_mask = nc_offsets < stride_nc
    spatial_mask = spatial_offsets < spatial_size
    
    # Initialize accumulators
    sum_x = tl.zeros((BLOCK_NC,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_NC,), dtype=tl.float32)
    
    # Process spatial elements for multiple (n,c) slices
    for s in range(0, BLOCK_SPATIAL, 128):
        spatial_idx = spatial_offsets + s
        spatial_chunk_mask = spatial_mask & (spatial_idx < spatial_size)
        
        # Check if any valid elements in this chunk
        has_valid = tl.sum(spatial_chunk_mask) > 0
        
        if has_valid:
            # Vectorized loading for better memory throughput
            for nc_idx in range(0, BLOCK_NC):
                if nc_idx < BLOCK_NC:
                    nc_idx_val = pid_nc + nc_idx
                    nc_valid = nc_idx_val < stride_nc
                    
                    if nc_valid:
                        x_ptrs = x_ptr + nc_idx_val * spatial_size + spatial_idx
                        x_chunk = tl.load(x_ptrs, mask=spatial_chunk_mask, other=0.0)
                        
                        # Update sums using masked operations
                        valid_count = tl.sum(spatial_chunk_mask)
                        if valid_count > 0:
                            # Create mask for this nc_idx
                            nc_idx_mask = tl.arange(0, BLOCK_NC) == nc_idx
                            x_sum = tl.sum(x_chunk)
                            x2_sum = tl.sum(x_chunk * x_chunk)
                            
                            # Update sums using where
                            sum_x = tl.where(
                                nc_idx_mask,
                                sum_x + x_sum,
                                sum_x
                            )
                            sum_x2 = tl.where(
                                nc_idx_mask,
                                sum_x2 + x2_sum,
                                sum_x2
                            )
    
    # Store partial results with proper masking
    if pid_spatial == 0:  # Only first spatial block stores
        tl.store(mean_ptr + nc_offsets, sum_x, mask=nc_mask)
        tl.store(var_ptr + nc_offsets, sum_x2, mask=nc_mask)


@triton.jit
def normalize_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    spatial_size,
    stride_nc,
    eps: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    """Normalize multiple (n,c) slices in parallel"""
    pid_spatial = tl.program_id(axis=0)
    pid_nc = tl.program_id(axis=1) * BLOCK_NC
    
    # Load mean and variance for this block of (n,c) slices
    nc_offsets = pid_nc + tl.arange(0, BLOCK_NC)
    nc_mask = nc_offsets < stride_nc
    
    mean = tl.load(mean_ptr + nc_offsets, mask=nc_mask, other=0.0)
    var = tl.load(var_ptr + nc_offsets, mask=nc_mask, other=0.0)
    
    # Compute mean and variance from sums
    mean_val = mean / spatial_size
    var_val = tl.maximum(var / spatial_size - mean_val * mean_val, 0.0)
    inv_std = tl.math.rsqrt(var_val + eps)
    
    # Process spatial elements
    spatial_offsets = pid_spatial * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    spatial_mask = spatial_offsets < spatial_size
    
    # Process in vectorized chunks
    for s in range(0, BLOCK_SPATIAL, 128):
        spatial_idx = spatial_offsets + s
        spatial_chunk_mask = spatial_mask & (spatial_idx < spatial_size)
        valid_count = tl.sum(spatial_chunk_mask)
        
        # Process each (n,c) slice
        for nc_idx in range(BLOCK_NC):
            nc_idx_val = pid_nc + nc_idx
            nc_valid = nc_idx_val < stride_nc
            
            if nc_valid and valid_count > 0:
                # Load input
                x_ptrs = x_ptr + nc_idx_val * spatial_size + spatial_idx
                x = tl.load(x_ptrs, mask=spatial_chunk_mask, other=0.0)
                
                # Get mean and inv_std for this nc_idx
                # Need to extract scalar values from arrays
                mean_scalar = tl.load(mean_ptr + nc_idx_val)
                var_scalar = tl.load(var_ptr + nc_idx_val)
                mean_val_scalar = mean_scalar / spatial_size
                var_val_scalar = tl.maximum(var_scalar / spatial_size - mean_val_scalar * mean_val_scalar, 0.0)
                inv_std_scalar = tl.math.rsqrt(var_val_scalar + eps)
                
                # Normalize
                normalized = (x - mean_val_scalar) * inv_std_scalar
                
                # Store output
                out_ptrs = output_ptr + nc_idx_val * spatial_size + spatial_idx
                tl.store(out_ptrs, normalized, mask=spatial_chunk_mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    N, C, H, W = x.shape
    spatial_size = H * W
    total_slices = N * C
    
    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Optimized block sizes for Ada Lovelace
    BLOCK_SPATIAL = 256
    BLOCK_NC = 8
    
    # Allocate buffers for moments
    mean = torch.zeros(total_slices, device=x.device, dtype=torch.float32)
    var = torch.zeros(total_slices, device=x.device, dtype=torch.float32)
    
    # Grid configuration for better SM utilization
    num_spatial_blocks = triton.cdiv(spatial_size, BLOCK_SPATIAL)
    num_nc_blocks = triton.cdiv(total_slices, BLOCK_NC)
    
    # Phase 1: Compute moments with optimized grid
    grid1 = (num_spatial_blocks, num_nc_blocks)
    compute_moments_kernel[grid1](
        x_contig, mean, var,
        spatial_size, total_slices,
        eps,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        BLOCK_NC=BLOCK_NC
    )
    
    # Phase 2: Normalize with same optimized grid
    grid2 = (num_spatial_blocks, num_nc_blocks)
    normalize_kernel[grid2](
        x_contig, output, mean, var,
        spatial_size, total_slices,
        eps,
        BLOCK_SPATIAL=BLOCK_SPATIAL,
        BLOCK_NC=BLOCK_NC
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x)
