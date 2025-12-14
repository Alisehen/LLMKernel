import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        # Small tensors: BLOCK_SIZE=256
        triton.Config({'BLOCK_SIZE': 256, 'USE_VECTORIZED_LOADS': False}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'USE_VECTORIZED_LOADS': False}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'USE_VECTORIZED_LOADS': False}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'USE_VECTORIZED_LOADS': False}, num_warps=8, num_stages=3),
        
        # Medium tensors: BLOCK_SIZE=512
        triton.Config({'BLOCK_SIZE': 512, 'USE_VECTORIZED_LOADS': False}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'USE_VECTORIZED_LOADS': False}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512, 'USE_VECTORIZED_LOADS': False}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'USE_VECTORIZED_LOADS': False}, num_warps=16, num_stages=4),
        
        # Large tensors: BLOCK_SIZE=1024, vectorized
        triton.Config({'BLOCK_SIZE': 1024, 'USE_VECTORIZED_LOADS': True}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'USE_VECTORIZED_LOADS': True}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'USE_VECTORIZED_LOADS': True}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'USE_VECTORIZED_LOADS': True}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'USE_VECTORIZED_LOADS': True}, num_warps=16, num_stages=5),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZED_LOADS: tl.constexpr,
):
    """Optimized GELU kernel with autotuned parameters for Ada Lovelace."""
    pid = tl.program_id(axis=0)
    
    if USE_VECTORIZED_LOADS:
        # Vectorized access: process 4 elements per thread
        block_start = pid * BLOCK_SIZE * 4
        offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)
        mask = offsets < n_elements
        
        # Load input with vectorized pattern
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Reshape for vector processing
        x_vec = tl.reshape(x, (BLOCK_SIZE, 4))
        
        # GELU constants
        a = 0.7978845608028654  # sqrt(2/π)
        b = 0.044715
        
        # Vectorized computation with fewer registers
        x3_vec = x_vec * x_vec * x_vec
        t_vec = a * (x_vec + b * x3_vec)
        
        # Optimized vectorized tanh approximation
        # Use 3-degree rational approximation for better accuracy vs performance
        t_sq_vec = t_vec * t_vec
        numerator_vec = t_vec * (27.0 + t_sq_vec)
        denominator_vec = 27.0 + 9.0 * t_sq_vec
        tanh_t_vec = numerator_vec / denominator_vec
        
        # Vectorized GELU output
        output_vec = 0.5 * x_vec * (1.0 + tanh_t_vec)
        
        # Reshape back and store
        output = tl.reshape(output_vec, (BLOCK_SIZE * 4,))
        tl.store(output_ptr + offsets, output, mask=mask)
        
    else:
        # Standard access pattern for smaller tensors
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Compute with optimized instruction scheduling
        x_sq = x * x
        x_cu = x_sq * x
        
        # GELU approximation
        t = 0.7978845608028654 * (x + 0.044715 * x_cu)
        
        # Optimized tanh approximation
        t_sq = t * t
        numerator = t * (27.0 + t_sq)
        denominator = 27.0 + 9.0 * t_sq
        tanh_t = numerator / denominator
        
        # Final GELU
        output = 0.5 * x * (1.0 + tanh_t)
        
        # Store result
        tl.store(output_ptr + offsets, output, mask=mask)

def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton-accelerated GELU activation with adaptive tuning."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Determine if vectorized loads should be used based on tensor size
    # For Ada Lovelace, vectorization helps more with larger tensors
    USE_VECTORIZED_LOADS = n_elements >= 65536
    
    # Set BLOCK_SIZE based on tensor size (autotune will override based on key)
    if n_elements < 4096:
        BLOCK_SIZE = 256
    elif n_elements < 262144:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    if USE_VECTORIZED_LOADS:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE * 4),)
    else:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with autotuning
    gelu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_VECTORIZED_LOADS=USE_VECTORIZED_LOADS,
    )
    
    return output

class ModelNew(nn.Module):
    """Optimized GELU implementation with autotuned Triton kernels."""
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)
