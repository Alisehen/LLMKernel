import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_stages': 2}, num_warps=32),
        triton.Config({'num_stages': 3}, num_warps=32),
        triton.Config({'num_stages': 4}, num_warps=32),
        triton.Config({'num_stages': 2}, num_warps=16),
        triton.Config({'num_stages': 3}, num_warps=16),
        triton.Config({'num_stages': 4}, num_warps=16),
        triton.Config({'num_stages': 2}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=8),
        triton.Config({'num_stages': 4}, num_warps=8),
    ],
    key=['n_elements', 'BLOCK_SIZE'],
)
@triton.jit
def softsign_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Optimized Softsign activation with autotuned num_stages.
    Uses direct computation: x / (1 + |x|) for maximum performance.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Direct computation: x / (1 + |x|)
    # This allows the compiler to optimize the operation pattern
    output = tl.math.fdiv(x, 1.0 + tl.abs(x))
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Triton wrapper for Softsign activation with optimized launch configuration.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Heuristic for block size based on tensor size
    if n_elements >= 262144:  # Large tensor
        BLOCK_SIZE = 1024
    elif n_elements >= 65536:   # Medium tensor
        BLOCK_SIZE = 512
    else:                       # Small tensor
        BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Set num_warps based on BLOCK_SIZE (must be BLOCK_SIZE // 32)
    num_warps = BLOCK_SIZE // 32
    
    # Launch optimized kernel with autotuned configuration
    softsign_kernel_optimized[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Softsign activation using autotuned Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softsign activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        # Use optimized Triton kernel with autotuning
        return triton_softsign(x)
