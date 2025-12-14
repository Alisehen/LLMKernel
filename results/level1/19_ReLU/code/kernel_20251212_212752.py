import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Optimized ReLU kernel with vectorized loads/stores and latency hiding."""
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create mask for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    
    # Fast ReLU: max(x, 0) = (x + abs(x)) / 2
    abs_x = tl.abs(x)
    output = (x + abs_x) * 0.5
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_relu_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized wrapper for ReLU activation with autotuned parameters."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Get optimal configuration
    if n_elements < 8192:
        config = {'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4, 'NUM_STAGES': 1, 'num_warps': 4}
    elif n_elements < 65536:
        config = {'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4, 'NUM_STAGES': 2, 'num_warps': 8}
    else:
        config = {'BLOCK_SIZE': 1024, 'VECTOR_SIZE': 8, 'NUM_STAGES': 3, 'num_warps': 16}
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    relu_kernel_optimized[grid](
        x, output, n_elements,
        BLOCK_SIZE=config['BLOCK_SIZE'],
        VECTOR_SIZE=config['VECTOR_SIZE'],
        NUM_STAGES=config['NUM_STAGES'],
        num_warps=config['num_warps']
    )
    
    return output


class ModelNew(nn.Module):
    """Optimized model using advanced Triton kernels for ReLU activation."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu_optimized(x)
