import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 8, 'num_stages': 4}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
    ],
    key=['n_elements'],
)
@triton.jit
def tanh_kernel(
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
    
    exp_2x = tl.exp(2.0 * x)
    output = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    output_flat = output.view(-1)
    x_flat = x.view(-1)
    
    n_elements = x_flat.numel()
    
    if n_elements >= 1048576:
        BLOCK_SIZE = 1024
    elif n_elements >= 262144:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    tanh_kernel[grid](x_flat, output_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)
