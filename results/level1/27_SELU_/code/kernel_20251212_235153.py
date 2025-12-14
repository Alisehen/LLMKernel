import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
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
    High-performance SELU kernel with optimized memory access and computation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU parameters (precomputed constants)
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946
    
    # Efficient SELU computation using mathematical transformations
    # For x > 0: SCALE * x
    # For x <= 0: SCALE * ALPHA * (exp(x) - 1)
    
    # Compute positive part
    pos_part = tl.maximum(x, 0.0)
    
    # Compute negative part efficiently
    # exp(x) for x <= 0, else 0
    exp_x = tl.exp(tl.minimum(x, 0.0))  # exp(0) = 1, but we'll mask it out for positive x
    neg_part = ALPHA * (exp_x - 1.0)
    # Only apply to non-positive values
    neg_part = tl.where(x <= 0.0, neg_part, 0.0)
    
    # Combine and scale
    result = SCALE * (pos_part + neg_part)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies SELU activation using optimized Triton kernel.
    
    Args:
        x: Input tensor of any shape.
        
    Returns:
        Output tensor with SELU applied.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Handle edge cases
    if n_elements == 0:
        return output
    
    # Use 1D grid with autotuned block size
    grid = (triton.cdiv(n_elements, 256),)  # Start with minimal grid
    
    # Launch kernel - autotuner will select optimal configuration
    selu_kernel[grid](
        x, output, n_elements
    )
    
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using optimized Triton kernels.
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
        return triton_selu(x)
