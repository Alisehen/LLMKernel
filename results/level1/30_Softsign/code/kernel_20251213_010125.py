import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softsign_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Optimized Softsign kernel with vectorized memory access.
    """
    pid = tl.program_id(axis=0)
    block_elems = BLOCK_SIZE * VECTOR_SIZE
    block_start = pid * block_elems
    
    # Vectorized offsets calculation
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
    offsets_flat = tl.reshape(offsets, block_elems)
    
    # Vectorized mask
    mask_flat = offsets_flat < n_elements
    mask = tl.reshape(mask_flat, (BLOCK_SIZE, VECTOR_SIZE))
    
    # Coalesced vector load
    x_vec = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute Softsign: x / (1 + |x|)
    # Using fused operations to reduce register pressure
    abs_x = tl.abs(x_vec)
    output_vec = x_vec / (1.0 + abs_x)
    
    # Coalesced vector store
    tl.store(output_ptr + offsets, output_vec, mask=mask)


@triton.autotune(
    configs=[
        # Large block size for maximum occupancy (FP32)
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 2, 'num_stages': 2},
                      num_warps=16),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 4, 'num_stages': 3},
                      num_warps=8),
        # Optimized for half precision
        triton.Config({'BLOCK_SIZE': 512, 'VECTOR_SIZE': 4, 'num_stages': 2},
                      num_warps=16),
        triton.Config({'BLOCK_SIZE': 256, 'VECTOR_SIZE': 8, 'num_stages': 3},
                      num_warps=8),
        # Small tensor configurations
        triton.Config({'BLOCK_SIZE': 128, 'VECTOR_SIZE': 4, 'num_stages': 1},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE': 64, 'VECTOR_SIZE': 8, 'num_stages': 1},
                      num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def softsign_kernel_autotuned(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Autotuned version with optimized configurations for different tensor sizes."""
    # Same implementation as above - Triton needs explicit copy for autotuning
    pid = tl.program_id(axis=0)
    block_elems = BLOCK_SIZE * VECTOR_SIZE
    block_start = pid * block_elems
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
    offsets_flat = tl.reshape(offsets, block_elems)
    
    mask_flat = offsets_flat < n_elements
    mask = tl.reshape(mask_flat, (BLOCK_SIZE, VECTOR_SIZE))
    
    x_vec = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    abs_x = tl.abs(x_vec)
    output_vec = x_vec / (1.0 + abs_x)
    
    tl.store(output_ptr + offsets, output_vec, mask=mask)


def triton_softsign(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized Triton wrapper for Softsign activation.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    if n_elements == 0:
        return output
    
    # Calculate optimal grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VECTOR_SIZE']),)
    
    # Launch autotuned kernel
    softsign_kernel_autotuned[grid](
        x, output, n_elements,
        num_stages=2  # Default, will be overridden by autotuner
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels with autotuning and vectorized memory access.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Softsign activation to the input tensor.
        """
        return triton_softsign(x)
