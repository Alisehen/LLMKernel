# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def tanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    x2 = x + x
    exp_2x = tl.exp(x2)
    y = (exp_2x - 1.0) / (exp_2x + 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    # Fallback for non-CUDA tensors (Triton requires GPU)
    if not x.is_cuda:
        return torch.tanh(x)

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()

    # Ensure grid size > 0 even for empty tensors
    def grid(meta):
        block = meta["BLOCK_SIZE"]
        return (triton.cdiv(max(n_elements, 1), block),)

    tanh_kernel[grid](
        x_contig,
        y,
        n_elements,
    )

    return y


class ModelNew(nn.Module):
    """
    Model that applies a high-performance Triton-based Tanh activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)
