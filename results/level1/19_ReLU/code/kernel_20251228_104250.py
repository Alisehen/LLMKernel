import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel_inplace(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC
    offsets = block_start + tl.arange(0, BLOCK_SIZE * VEC)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(x_ptr + offsets, x, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance in-place-style ReLU using Triton.
    Operates on a contiguous CUDA tensor buffer to maximize throughput.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    # Ensure a contiguous buffer for coalesced memory access.
    # If x is already contiguous, this is a cheap view.
    x_contig = x.contiguous()

    n_elements = x_contig.numel()
    BLOCK_SIZE = 256  # power-of-2
    VEC = 4           # process 4*BLOCK_SIZE elements per program

    def grid(meta):
        return (triton.cdiv(max(n_elements, 1), meta["BLOCK_SIZE"] * meta["VEC"]),)

    relu_kernel_inplace[grid](
        x_contig,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC=VEC,
        num_warps=8,
    )
    # Same values as torch.relu(x), with x_contig updated in-place.
    return x_contig.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs a ReLU activation.
    Uses an in-place Triton kernel on a contiguous buffer to minimize memory traffic.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
