import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,          # *const T
    y_ptr,          # *T
    n_elements,     # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous memory for flat indexing
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    relu_kernel[grid](
        x_contig,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Model using a high-performance Triton ReLU kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x to be on CUDA for Triton execution
        return triton_relu(x)
