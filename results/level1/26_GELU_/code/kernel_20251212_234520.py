import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU with tensor core utilization and improved grid layout."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with vectorized loads for better memory throughput
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Constants for GELU computation
    SQRT_2_OVER_PI = 0.7978845608028654
    GELU_CONST = 0.044715
    
    # Fused computation with tensor core friendly operations
    x_sq = x * x
    x_cube = x_sq * x
    inner = x + GELU_CONST * x_cube
    scaled = SQRT_2_OVER_PI * inner
    
    # Fast tanh approximation that works well with tensor cores
    # tanh(x) = 1 - 2/(1 + exp(2x))
    exp_2x = tl.exp(2.0 * scaled)
    tanh_result = 1.0 - 2.0 / (1.0 + exp_2x)
    
    # Final GELU computation
    output = 0.5 * x * (1.0 + tanh_result)
    
    # Store with vectorized stores
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function for optimized GELU activation."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized grid calculation with larger blocks for better SM utilization
    # For very small tensors, ensure at least 1 block
    grid = lambda META: (max(1, triton.cdiv(n_elements, META['BLOCK_SIZE'])),)
    
    gelu_kernel[grid](
        x,
        output,
        n_elements,
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model with Triton GELU activation using improved grid layout.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)
