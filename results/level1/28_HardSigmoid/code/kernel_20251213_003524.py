import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=16, num_stages=3),  # Original
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=8, num_stages=4),   # +1 stage
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=8, num_stages=2),   # -1 stage
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=3),   # Smaller warp
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=2, num_stages=3),   # Smallest warp
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4}, num_warps=16, num_stages=4),  # Max warp, +1 stage
    ],
    key=['n_elements']
)
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
    
    # HardSigmoid computation with vector operations
    one_sixth = 0.1666666716337204  # 1/6 as float32
    half = 0.5
    
    # Vectorized FMA operations
    linear = x_vector * one_sixth + half
    clamped = tl.minimum(tl.maximum(linear, 0.0), 1.0)
    
    # Vectorized store
    tl.store(output_ptr + offsets, clamped, mask=mask)

def triton_hardsigmoid_optimized(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Grid calculation optimized for vectorization
    def grid(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE'] * META['VECTOR_SIZE']),)
    
    # Kernel launch with autotuning
    hardsigmoid_kernel_optimized[grid](x, output, n_elements)
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid_optimized(x)
