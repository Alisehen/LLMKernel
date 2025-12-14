import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute softplus(x) = log(1 + exp(x))
    # Use log1p for better numerical stability
    softplus_val = tl.log1p(tl.exp(x))
    
    # Compute tanh(softplus(x))
    tanh_val = tl.tanh(softplus_val)
    
    # Multiply with original x: x * tanh(softplus(x))
    output = x * tanh_val
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Use optimal block size for A100/A10
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_activation_kernel[grid](
        x, output, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused activation using Triton kernel
        x = triton_fused_activation(x)
        
        # Apply batch normalization
        x = self.bn(x)
        return x
