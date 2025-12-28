# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardsigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise HardSigmoid:
        y = max(0, min(1, (x + 3) / 6))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # (x + 3) / 6
    y = (x + 3.0) * (1.0 / 6.0)
    # clamp to [0, 1]
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton HardSigmoid implementation.
    Falls back to PyTorch if input is not CUDA.
    """
    if not x.is_cuda:
        # CPU fallback: preserve original behavior
        return nn.functional.hardsigmoid(x)

    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    x_flat = x_contig.view(-1)
    out_flat = out.view(-1)
    n_elements = x_flat.numel()

    BLOCK_SIZE = 256  # power-of-2 as required
    grid = lambda meta: (max(1, triton.cdiv(n_elements, meta["BLOCK_SIZE"])),)

    hardsigmoid_kernel[grid](
        x_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,  # balanced for BLOCK_SIZE=256
    )

    return out.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs a HardSigmoid activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)
