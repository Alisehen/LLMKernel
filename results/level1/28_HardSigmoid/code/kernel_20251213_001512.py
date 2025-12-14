import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardsigmoid_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VECTOR_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE * VECTOR_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load
    x_vector = tl.load(x_ptr + offsets, mask=mask)
    
    # Reshape to (BLOCK_SIZE, VECTOR_SIZE) for efficient computation
    x = tl.reshape(x_vector, (BLOCK_SIZE, VECTOR_SIZE))
    
    # HardSigmoid computation with vector operations
    one_sixth = 0.1666666716337204  # 1/6 as float32
    half = 0.5
    
    # Vectorized FMA operations
    linear = x * one_sixth + half
    clamped = tl.minimum(tl.maximum(linear, 0.0), 1.0)
    
    # Vectorized store
    clamped_vector = tl.reshape(clamped, (BLOCK_SIZE * VECTOR_SIZE,))
    tl.store(output_ptr + offsets, clamped_vector, mask=mask)

def triton_hardsigmoid_optimized(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Autotune configurations for optimal performance
    configs = [
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 4}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 2}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 2}, num_warps=32),
    ]
    
    # Grid calculation optimized for vectorization
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE'] * META['VECTOR_SIZE']),)
    
    # Kernel launch with autotuning
    kernel = triton.autotune(configs, key=['n_elements'])
    kernel[grid](x, output, n_elements)
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid_optimized(x)
