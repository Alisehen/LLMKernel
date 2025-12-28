import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tile, good for small/irregular sizes
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        # Larger tile, higher occupancy / bandwidth on 4090
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        # Same tile with more warps to push SM utilization
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def sigmoid_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    y = 1.0 / (1.0 + tl.exp(-x))

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton implementation of torch.sigmoid(x).
    """
    assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()

    def grid(meta):
        block_size = meta["BLOCK_SIZE"]
        # Ensure grid size > 0 even for empty tensors
        return (max(1, triton.cdiv(n_elements, block_size)),)

    sigmoid_kernel[grid](
        x_contig,
        y,
        n_elements,
    )

    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Model that performs a Sigmoid activation using an optimized Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)
