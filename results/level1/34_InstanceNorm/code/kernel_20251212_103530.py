import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_moments_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    n_elements,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum and sum of squares for this block
    sum_x = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)
    
    # Store block results
    tl.store(mean_ptr + pid, sum_x)
    tl.store(var_ptr + pid, sum_x2)


@triton.jit
def reduce_moments_kernel(
    mean_ptr,
    var_ptr,
    final_mean_ptr,
    final_var_ptr,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Load partial sums
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.cdiv(spatial_size, BLOCK_SIZE)
    
    mean_partial = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    var_partial = tl.load(var_ptr + offsets, mask=mask, other=0.0)
    
    # Final reduction
    sum_x = tl.sum(mean_partial)
    sum_x2 = tl.sum(var_partial)
    
    mean_val = sum_x / spatial_size
    var_val = sum_x2 / spatial_size - mean_val * mean_val
    
    # Store final results
    tl.store(final_mean_ptr + pid, mean_val)
    tl.store(final_var_ptr + pid, var_val)


@triton.jit
def normalize_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_spatial = tl.program_id(axis=1)
    
    # Load mean and variance for this (n,c) slice
    mean = tl.load(mean_ptr + pid_nc)
    var = tl.load(var_ptr + pid_nc)
    
    # Compute normalized values
    inv_std = tl.math.rsqrt(var + eps)
    
    # Process spatial elements
    spatial_start = pid_spatial * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = offsets < spatial_size
    
    # Load input for this spatial block
    x_offset = pid_nc * spatial_size + offsets
    x = tl.load(x_ptr + x_offset, mask=spatial_mask)
    
    # Normalize
    normalized = (x - mean) * inv_std
    
    # Store output
    tl.store(output_ptr + x_offset, normalized, mask=spatial_mask)


def triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    N, C, H, W = x.shape
    spatial_size = H * W
    total_slices = N * C
    
    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Allocate buffers for moments
    BLOCK_SIZE = 1024
    num_spatial_blocks = triton.cdiv(spatial_size, BLOCK_SIZE)
    
    # Phase 1: Compute block-wise moments
    mean_partial = torch.zeros(total_slices * num_spatial_blocks, device=x.device, dtype=x.dtype)
    var_partial = torch.zeros(total_slices * num_spatial_blocks, device=x.device, dtype=x.dtype)
    
    grid1 = lambda meta: (total_slices * num_spatial_blocks,)
    compute_moments_kernel[grid1](
        x_contig, mean_partial, var_partial,
        total_slices * spatial_size, spatial_size,
        eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Reduce moments for each (n,c) slice
    final_mean = torch.zeros(total_slices, device=x.device, dtype=x.dtype)
    final_var = torch.zeros(total_slices, device=x.device, dtype=x.dtype)
    
    grid2 = lambda meta: (total_slices,)
    reduce_moments_kernel[grid2](
        mean_partial, var_partial, final_mean, final_var,
        spatial_size, eps, BLOCK_SIZE=num_spatial_blocks
    )
    
    # Phase 3: Apply normalization
    num_blocks_spatial = triton.cdiv(spatial_size, BLOCK_SIZE)
    grid3 = lambda meta: (total_slices, num_blocks_spatial)
    normalize_kernel[grid3](
        x_contig, output, final_mean, final_var,
        spatial_size, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm(x)
