import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Mish activation: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    tanh_softplus = tl.tanh(softplus)
    mish_out = x * tanh_softplus
    
    tl.store(output_ptr + offsets, mish_out, mask=mask)

@triton.jit
def double_mish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # First Mish
    softplus1 = tl.log(1.0 + tl.exp(x))
    tanh_softplus1 = tl.tanh(softplus1)
    y = x * tanh_softplus1
    
    # Second Mish
    softplus2 = tl.log(1.0 + tl.exp(y))
    tanh_softplus2 = tl.tanh(softplus2)
    z = y * tanh_softplus2
    
    tl.store(output_ptr + offsets, z, mask=mask)

def triton_mish(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_double_mish(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    double_mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    Uses fused Triton kernels for the Mish activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Use fused double mish kernel instead of two separate calls
        x = triton_double_mish(x)
        return x
