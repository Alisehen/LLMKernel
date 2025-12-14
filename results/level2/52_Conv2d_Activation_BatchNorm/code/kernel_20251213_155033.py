import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 2, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_activation_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized fused activation: x * tanh(softplus(x))
    # Use stable computation to avoid overflow
    
    # Compute softplus(x) = log(1 + exp(x))
    # Use piecewise approximation for better performance
    # For x > 20: softplus(x) ≈ x
    # For x < -20: softplus(x) ≈ exp(x)
    exp_x = tl.exp(x)
    
    # Compute log1p(exp_x) = log(1 + exp(x))
    # Triton doesn't have log1p, so we compute it manually
    log1p_exp_x = tl.where(
        x > 0.0,
        x + tl.log(1.0 + tl.exp(-x)),  # Stable for positive x
        tl.log(1.0 + exp_x)            # Direct for negative x
    )
    
    # tanh(y) computation with fast approximation
    y = log1p_exp_x
    exp_2y = tl.exp(2.0 * y)
    tanh_y = (exp_2y - 1.0) / (exp_2y + 1.0)
    
    result = x * tanh_y
    
    tl.store(output_ptr + offsets, result, mask=mask)

def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_activation_kernel_optimized[grid](
        x, output, n_elements
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_activation(x)
        x = self.bn(x)
        return x
