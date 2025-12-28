# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "VEC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256, "VEC": 4}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    """
    Vectorized, bandwidth-oriented HardSigmoid:
        y = clamp(x / 6 + 0.5, 0, 1)
    Each program processes BLOCK_SIZE * VEC contiguous elements.
    """
    # Each program handles this many elements
    elements_per_program = BLOCK_SIZE * VEC

    pid = tl.program_id(axis=0)
    block_start = pid * elements_per_program

    # tl.arange bounds must be constexpr: use BLOCK_SIZE * VEC directly
    offsets = block_start + tl.arange(0, BLOCK_SIZE * VEC)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # PyTorch HardSigmoid: clamp(x / 6 + 0.5, 0, 1)
    y = x * (1.0 / 6.0) + 0.5
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies HardSigmoid using a Triton kernel when possible, otherwise
    falls back to PyTorch's implementation for unsupported inputs/devices.
    """
    if (not x.is_cuda) or (not x.is_contiguous()):
        return torch.nn.functional.hardsigmoid(x)

    y = torch.empty_like(x)
    n_elements = x.numel()

    # Each program handles BLOCK_SIZE * VEC elements (determined by autotune).
    def grid(meta):
        elements_per_program = meta["BLOCK_SIZE"] * meta["VEC"]
        return (max(1, triton.cdiv(n_elements, elements_per_program)),)

    hardsigmoid_kernel[grid](
        x,
        y,
        n_elements,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated HardSigmoid model with a bandwidth-optimized kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact behavioral match to the reference:
        # return torch.nn.functional.hardsigmoid(x)
        return triton_hardsigmoid(x)
