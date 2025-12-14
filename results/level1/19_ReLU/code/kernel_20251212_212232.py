import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'ELEMENTS_PER_THREAD': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'ELEMENTS_PER_THREAD': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'ELEMENTS_PER_THREAD': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'ELEMENTS_PER_THREAD': 16}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    """Optimized ReLU kernel with multiple elements per thread and better grid utilization."""
    pid = tl.program_id(axis=0)
    
    # Each block processes BLOCK_SIZE * ELEMENTS_PER_THREAD elements
    block_start = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Process multiple elements per thread
    for i in range(ELEMENTS_PER_THREAD):
        cur_offsets = offsets + i * BLOCK_SIZE
        mask = cur_offsets < n_elements
        
        if tl.sum(mask) > 0:  # Only process if any elements are valid
            x = tl.load(x_ptr + cur_offsets, mask=mask)
            # Fast ReLU using arithmetic: (x + |x|) / 2
            abs_x = tl.abs(x)
            output = (x + abs_x) * 0.5
            tl.store(output_ptr + cur_offsets, output, mask=mask)


def triton_relu_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton wrapper for ReLU activation with autotuning."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    if n_elements == 0:
        return output
    
    # Use autotuned grid calculation
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE'] * META['ELEMENTS_PER_THREAD']),)
    
    relu_kernel_optimized[grid](
        x, output, n_elements,
    )
    
    return output


class ModelNew(nn.Module):
    """Optimized model using advanced Triton kernels for ReLU activation."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu_optimized(x)
