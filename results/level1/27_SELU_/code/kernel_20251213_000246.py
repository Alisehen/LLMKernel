import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Optimized configurations based on NCU metrics
        # High DRAM utilization suggests we need better memory latency hiding
        # L1 hit rate of 50% suggests we can improve caching behavior
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),  # Lower stages for small blocks
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),  # Increased stages for latency hiding
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),  # More warps for larger blocks
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=5),  # Extreme block size for maximum reuse
    ],
    key=['n_elements'],
)
@triton.jit
def selu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ultra-optimized SELU kernel with improved memory access patterns and latency hiding.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Use vectorized offsets for better memory coalescing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with cache hints for better L1 utilization
    # 'nc' = non-coherent, reduces cache pollution for streaming data
    x = tl.load(x_ptr + offsets, mask=mask, cache_modifier='.nc')
    
    # SELU parameters
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946
    
    # Optimized branch-free computation with early materialization
    # This formulation reduces register pressure and improves ILP
    neg_mask = x <= 0.0
    exp_x_minus_one = tl.exp(x) - 1.0
    result = SCALE * tl.where(neg_mask, ALPHA * exp_x_minus_one, x)
    
    # Store with cache modifier
    tl.store(output_ptr + offsets, result, mask=mask, cache_modifier='.nc')


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies SELU activation using ultra-optimized Triton kernel.
    
    Args:
        x: Input tensor of any shape.
        
    Returns:
        Output tensor with SELU applied.
    """
    # Ensure contiguous memory layout
    x_flat = x.flatten().contiguous()
    n_elements = x_flat.numel()
    
    if n_elements == 0:
        return x
    
    # Allocate output
    output = torch.empty_like(x_flat)
    
    # Calculate optimal grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with optimized configuration
    selu_kernel[grid](
        x_flat, output, n_elements
    )
    
    # Reshape back to original shape
    return output.view_as(x)


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using ultra-optimized Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous()
        return triton_selu(x)
