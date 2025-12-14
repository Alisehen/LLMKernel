import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_instance_norm_kernel(
    x_ptr,
    output_ptr,
    spatial_size,
    total_slices,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Fused kernel that computes mean, variance, and normalization in a single pass."""
    pid = tl.program_id(0)  # which (n,c) slice
    
    if pid >= total_slices:
        return
    
    # Initialize accumulators
    sum_x = tl.zeros((VECTOR_SIZE,), tl.float32)
    sum_x2 = tl.zeros((VECTOR_SIZE,), tl.float32)
    count = 0
    
    # Process in vectorized blocks
    for off in range(0, spatial_size, BLOCK_SIZE * VECTOR_SIZE):
        # Create vectorized indices
        idx = off + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
        idx_flat = idx.flatten()
        mask_flat = idx_flat < spatial_size
        
        # Vectorized load
        x_vec = tl.load(
            x_ptr + pid * spatial_size + idx_flat,
            mask=mask_flat,
            other=0.0,
            cache_modifier=".cg",  # Cache global to reduce L1 pressure
        )
        
        # Reshape and accumulate
        x_vec_reshaped = tl.reshape(x_vec, (BLOCK_SIZE, VECTOR_SIZE))
        valid_mask = tl.reshape(mask_flat, (BLOCK_SIZE, VECTOR_SIZE))
        
        for v in range(VECTOR_SIZE):
            if tl.sum(valid_mask[:, v]) > 0:
                sum_x = tl.where(valid_mask[:, v], sum_x + tl.sum(x_vec_reshaped[:, v]), sum_x)
                sum_x2 = tl.where(valid_mask[:, v], sum_x2 + tl.sum(x_vec_reshaped[:, v] * x_vec_reshaped[:, v]), sum_x2)
                count += tl.sum(valid_mask[:, v])
    
    # Final reduction across vector lanes
    final_sum_x = tl.sum(sum_x)
    final_sum_x2 = tl.sum(sum_x2)
    
    # Compute mean and variance
    mean_val = final_sum_x / count
    variance = tl.maximum(final_sum_x2 / count - mean_val * mean_val, 0.0)
    inv_std = tl.math.rsqrt(variance + eps)
    
    # Normalization pass with better caching
    for off in range(0, spatial_size, BLOCK_SIZE * VECTOR_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
        idx_flat = idx.flatten()
        mask_flat = idx_flat < spatial_size
        
        # Load again but with caching hint for temporal locality
        x_vec = tl.load(
            x_ptr + pid * spatial_size + idx_flat,
            mask=mask_flat,
            other=0.0,
            cache_modifier=".ca",  # Cache all to keep in L1
        )
        
        # Normalize
        normalized = (x_vec - mean_val) * inv_std
        
        # Store
        tl.store(
            output_ptr + pid * spatial_size + idx_flat,
            normalized,
            mask=mask_flat,
            cache_modifier=".wb",  # Write-back policy
        )


def optimized_triton_instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    N, C, H, W = x.shape
    spatial_size = H * W
    total_slices = N * C
    
    # Ensure contiguous memory layout
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Autotune configurations
    configs = [
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4, 'num_stages': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4, 'num_stages': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 2, 'num_stages': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 2, 'num_stages': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 2, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 1, 'num_stages': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 1, 'num_stages': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 1, 'num_stages': 3}, num_warps=8),
    ]
    
    @triton.autotune(configs=configs, key=['spatial_size', 'total_slices'])
    @triton.jit
    def tuned_kernel(
        x_ptr, output_ptr, spatial_size, total_slices, eps,
        BLOCK_SIZE: tl.constexpr, VECTOR_SIZE: tl.constexpr, num_stages: tl.constexpr
    ):
        fused_instance_norm_kernel(
            x_ptr, output_ptr, spatial_size, total_slices, eps,
            BLOCK_SIZE, VECTOR_SIZE, num_stages
        )
    
    # Launch optimized kernel
    grid = (total_slices,)
    tuned_kernel[grid](
        x_contig, output, spatial_size, total_slices, eps
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return optimized_triton_instance_norm(x)
