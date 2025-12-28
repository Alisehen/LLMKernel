import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC
    offsets = block_start + tl.arange(0, BLOCK_SIZE * VEC)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance ReLU using Triton.
    Works on contiguous CUDA tensors of any shape.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = y.numel()
    BLOCK_SIZE = 256  # power-of-2 as required
    VEC = 4           # process 4*BLOCK_SIZE elements per program for higher throughput

    def grid(meta):
        # ensure grid size > 0
        return (triton.cdiv(max(n_elements, 1), meta["BLOCK_SIZE"] * meta["VEC"]),)

    relu_kernel[grid](
        x_contig,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        num_warps=8,
    )
    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
