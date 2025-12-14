import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_stages': 2}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=8),
        triton.Config({'num_stages': 4}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=4),
        triton.Config({'num_stages': 3}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def double_mish_kernel_optimized(
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
    
    # First Mish: x * tanh(softplus(x))
    # Compute softplus as log1p(exp(x)) for numerical stability
    x_exp = tl.exp(x)
    softplus1 = tl.log1p(x_exp)
    exp_2s1 = tl.exp(2.0 * softplus1)
    tanh_softplus1 = (exp_2s1 - 1.0) / (exp_2s1 + 1.0)
    y = x * tanh_softplus1
    
    # Second Mish
    y_exp = tl.exp(y)
    softplus2 = tl.log1p(y_exp)
    exp_2s2 = tl.exp(2.0 * softplus2)
    tanh_softplus2 = (exp_2s2 - 1.0) / (exp_2s2 + 1.0)
    z = y * tanh_softplus2
    
    tl.store(output_ptr + offsets, z, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'num_stages': 2}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=8),
        triton.Config({'num_stages': 4}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def mish_kernel_optimized(
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
    
    # Optimized single Mish
    x_exp = tl.exp(x)
    softplus = tl.log1p(x_exp)
    exp_2s = tl.exp(2.0 * softplus)
    tanh_softplus = (exp_2s - 1.0) / (exp_2s + 1.0)
    mish_out = x * tanh_softplus
    
    tl.store(output_ptr + offsets, mish_out, mask=mask)

def triton_mish_optimized(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mish_kernel_optimized[grid](
        x, output, n_elements
    )
    return output

def triton_double_mish_optimized(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    double_mish_kernel_optimized[grid](
        x, output, n_elements
    )
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
        x = triton_double_mish_optimized(x)
        return x
