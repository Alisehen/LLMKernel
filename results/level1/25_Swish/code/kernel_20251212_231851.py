import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4}, num_stages=2, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def swish_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """Optimized Swish activation with vectorized memory operations."""
    pid = tl.program_id(axis=0)
    elements_per_block = BLOCK_SIZE * VECTOR_SIZE
    block_start = pid * elements_per_block
    
    # Vectorized offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
    offsets = tl.ravel(offsets)
    mask = offsets < n_elements
    
    # Vectorized load
    x_vec = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Stable swish computation with vectorized operations
    abs_x = tl.abs(x_vec)
    exp_neg_abs_x = tl.exp(-abs_x)
    denom = 1.0 + exp_neg_abs_x
    numerator = tl.where(x_vec >= 0, 1.0, exp_neg_abs_x)
    sigmoid_x = numerator / denom
    output_vec = x_vec * sigmoid_x
    
    # Vectorized store
    tl.store(output_ptr + offsets, output_vec, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_stages=3, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def swish_fast_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """Optimized fast swish with vectorized memory operations."""
    pid = tl.program_id(axis=0)
    elements_per_block = BLOCK_SIZE * VECTOR_SIZE
    block_start = pid * elements_per_block
    
    # Vectorized offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
    offsets = tl.ravel(offsets)
    mask = offsets < n_elements
    
    # Vectorized load
    x_vec = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fast approximation with vectorized operations
    half_x = x_vec * 0.5
    # Clipped exponential for stability
    exp_input = tl.where(half_x > 10.0, 10.0, tl.where(half_x < -10.0, -10.0, half_x))
    exp_x = tl.exp(exp_input)
    tanh_half_x = (exp_x - 1.0) / (exp_x + 1.0)
    sigmoid_approx = 0.5 + 0.5 * tanh_half_x
    output_vec = x_vec * sigmoid_approx
    
    # Vectorized store
    tl.store(output_ptr + offsets, output_vec, mask=mask)

def triton_swish_optimized(x: torch.Tensor, fast_math: bool = False) -> torch.Tensor:
    """Optimized Triton Swish with autotuned configurations."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Autotuned configurations based on input size and memory patterns
    if n_elements <= 8192:
        BLOCK_SIZE = 128
        VECTOR_SIZE = 4
    elif n_elements <= 65536:
        BLOCK_SIZE = 256
        VECTOR_SIZE = 4
    elif n_elements <= 262144:
        BLOCK_SIZE = 512
        VECTOR_SIZE = 2
    else:
        # Large tensors: maximize occupancy and hide memory latency
        BLOCK_SIZE = 1024
        VECTOR_SIZE = 2
    
    elements_per_block = BLOCK_SIZE * VECTOR_SIZE
    grid = (triton.cdiv(n_elements, elements_per_block),)
    
    if fast_math:
        swish_fast_kernel_optimized[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            VECTOR_SIZE=VECTOR_SIZE,
        )
    else:
        swish_kernel_optimized[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            VECTOR_SIZE=VECTOR_SIZE,
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using advanced Triton kernels.
    """
    def __init__(self, use_fast_math: bool = False):
        super(ModelNew, self).__init__()
        self.use_fast_math = use_fast_math
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        
        return triton_swish_optimized(x, self.use_fast_math)
