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
    total_slices,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and variance for each (n,c) slice using a single thread per slice"""
    pid = tl.program_id(0)  # which (n,c) slice
    
    if pid >= total_slices:
        return
    
    # Initialize accumulators
    sum_x = 0.0
    sum_x2 = 0.0
    
    # Process the entire spatial dimension in blocks
    for off in range(0, spatial_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < spatial_size
        
        # Load data for this block
        x = tl.load(x_ptr + pid * spatial_size + idx, mask=mask, other=0.0)
        
        # Update accumulators
        sum_x += tl.sum(x)
        sum_x2 += tl.sum(x * x)
    
    # Compute final mean and variance
    mean_val = sum_x / spatial_size
    variance = tl.maximum(sum_x2 / spatial_size - mean_val * mean_val, 0.0)
    
    # Store results
    tl.store(mean_ptr + pid, mean_val)
    tl.store(var_ptr + pid, variance)


@triton.jit
def normalize_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    spatial_size,
    total_slices,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize each (n,c) slice using the precomputed mean and variance"""
    pid = tl.program_id(0)  # which (n,c) slice
    
    if pid >= total_slices:
        return
    
    # Load mean and variance for this slice
    mean_val = tl.load(mean_ptr + pid)
    var_val = tl.load(var_ptr + pid)
    
    # Compute inverse standard deviation
    inv_std = tl.math.rsqrt(var_val + eps)
    
    # Process the entire spatial dimension in blocks
    for off in range(0, spatial_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < spatial_size
        
        # Load input data
        x = tl.load(x_ptr + pid * spatial_size + idx, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x - mean_val) * inv_std
        
        # Store output
        tl.store(output_ptr + pid * spatial_size + idx, normalized, mask=mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    N, C, H, W = x.shape
    spatial_size = H * W
    total_slices = N * C
    
    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Optimized block size for spatial dimension processing
    BLOCK_SIZE = 1024
    
    # Allocate buffers for moments
    mean = torch.zeros(total_slices, device=x.device, dtype=torch.float32)
    var = torch.zeros(total_slices, device=x.device, dtype=torch.float32)
    
    # Launch compute moments kernel
    grid = (total_slices,)
    compute_moments_kernel[grid](
        x_contig, mean, var,
        spatial_size, total_slices,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch normalize kernel
    normalize_kernel[grid](
        x_contig, output, mean, var,
        spatial_size, total_slices,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x)
