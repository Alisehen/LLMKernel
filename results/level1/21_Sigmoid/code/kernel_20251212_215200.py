import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid of blocks
    pid = tl.program_id(axis=0)
    
    # Compute block start and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checks
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Use exp2 for better precision and faster computation
    neg_x = -x
    # Compute exp(-x) using exp2 for stability
    exp_neg_x = tl.exp(neg_x)
    denominator = 1.0 + exp_neg_x
    output = 1.0 / denominator
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Preserve input shape
    output = torch.empty_like(x)
    
    # Flatten for 1D kernel
    x_flatten = x.flatten()
    output_flatten = output.flatten()
    
    # Compute kernel parameters
    n_elements = x_flatten.numel()
    if n_elements == 0:
        return output
    
    # Optimal block size: 1024 for maximum occupancy on target hardware
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    sigmoid_kernel[grid](
        x_flatten, output_flatten, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs Sigmoid activation using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using Triton kernel.
        """
        return triton_sigmoid(x)
