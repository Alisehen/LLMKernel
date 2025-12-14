import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=32),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=32),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=32),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.minimum(tl.maximum(x, -1.0), 1.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    if not x.is_contiguous():
        x = x.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardtanh_kernel[grid](x, output, n_elements)
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardtanh(x)
