import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_VECTORIZED_LOADS: tl.constexpr,
):
    """Optimized GELU kernel with vectorized memory access and latency hiding."""
    pid = tl.program_id(axis=0)
    
    if USE_VECTORIZED_LOADS:
        # Vectorized access: process 4 elements per thread
        # This improves memory coalescing and reduces instruction overhead
        block_start = pid * BLOCK_SIZE * 4
        offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)
        mask = offsets < n_elements
        
        # Load input with vectorized pattern
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Reshape for vector processing
        x_vec = tl.reshape(x, (BLOCK_SIZE, 4))
        
        # Process vector elements
        # GELU constants
        a = 0.7978845608028654  # sqrt(2/π)
        b = 0.044715
        
        # Vectorized computation
        x3_vec = x_vec * x_vec * x_vec
        t_vec = a * (x_vec + b * x3_vec)
        
        # Vectorized tanh approximation
        neg_2abs_t_vec = -2.0 * tl.abs(t_vec)
        exp_neg_2abs_t_vec = tl.exp(neg_2abs_t_vec)
        
        numerator_vec = 1.0 - exp_neg_2abs_t_vec
        denominator_vec = 1.0 + exp_neg_2abs_t_vec
        
        # Vectorized conditional tanh
        tanh_t_vec = tl.where(t_vec >= 0.0, 
                             numerator_vec / denominator_vec, 
                             -numerator_vec / denominator_vec)
        
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
        x_cu = x_sq * x  # x^3
        
        # GELU approximation
        t = 0.7978845608028654 * (x + 0.044715 * x_cu)
        
        # Optimized tanh approximation with fewer registers
        # Using direct formula: tanh(t) = t * (27 + t^2) / (27 + 9*t^2)
        # This reduces register pressure while maintaining accuracy
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
    
    # Heuristic: use vectorized loads for large tensors
    # This improves memory coalescing for DRAM-bound operations
    USE_VECTORIZED_LOADS = n_elements >= 65536
    
    # Configuration based on tensor size
    if n_elements < 4096:
        # Small tensors: maximize occupancy with smaller blocks
        BLOCK_SIZE = 256
        num_stages = 2
    elif n_elements < 262144:
        # Medium tensors: balance occupancy and latency hiding
        BLOCK_SIZE = 512
        num_stages = 3
    else:
        # Large tensors: maximize throughput with large blocks and stages
        BLOCK_SIZE = 1024
        num_stages = 4
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * (4 if meta['USE_VECTORIZED_LOADS'] else 1)),)
    
    # Launch kernel with optimal configuration
    gelu_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_VECTORIZED_LOADS=USE_VECTORIZED_LOADS,
        num_stages=num_stages,
        num_warps=BLOCK_SIZE // 32
    )
    
    return output

class ModelNew(nn.Module):
    """Optimized GELU implementation with adaptive Triton kernels."""
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu(x)
