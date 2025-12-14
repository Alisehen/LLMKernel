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
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SELU parameters
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946
    
    # Compute SELU using branch-free formulation
    # For x > 0: SCALE * x
    # For x <= 0: SCALE * ALPHA * (exp(x) - 1)
    
    # Efficient computation: use where to avoid branches
    is_pos = x > 0.0
    pos_part = SCALE * x
    neg_part = SCALE * ALPHA * (tl.exp(x) - 1.0)
    result = tl.where(is_pos, pos_part, neg_part)
    
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
    # Ensure contiguous memory layout
    x_flat = x.flatten().contiguous()
    n_elements = x_flat.numel()
    
    if n_elements == 0:
        return x
    
    # Allocate output
    output = torch.empty_like(x_flat)
    
    # Calculate grid with minimum blocks
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    selu_kernel[grid](
        x_flat, output, n_elements
    )
    
    # Reshape back to original shape
    return output.view_as(x)


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
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous()
        return triton_selu(x)
