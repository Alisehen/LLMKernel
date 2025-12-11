import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # LeakyReLU: x if x >= 0 else negative_slope * x
    condition = x >= 0.0
    result = tl.where(condition, x, x * negative_slope)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimal BLOCK_SIZE for elementwise ops on A100
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    leaky_relu_kernel[grid](
        x,
        output,
        n_elements,
        negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_leaky_relu(x, self.negative_slope)
