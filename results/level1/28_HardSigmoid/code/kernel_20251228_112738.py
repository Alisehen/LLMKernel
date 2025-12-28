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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # PyTorch HardSigmoid: clamp(x / 6 + 0.5, 0, 1)
    y = x * (1.0 / 6.0) + 0.5
    y = tl.maximum(y, 0.0)
    y = tl.minimum(y, 1.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    # Expect contiguous CUDA tensor for Triton
    if not x.is_cuda or not x.is_contiguous():
        # Fallback to PyTorch implementation on unsupported inputs/devices
        return nn.functional.hardsigmoid(x)

    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256  # power-of-2, good occupancy

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardsigmoid_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated HardSigmoid model.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)
