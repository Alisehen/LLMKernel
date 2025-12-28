import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_kernel(
    A_ptr,          # *const T
    C_ptr,          # *mut T
    s,              # scalar multiplier
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    c = a * s
    tl.store(C_ptr + offsets, c, mask=mask)


def triton_scale(A: torch.Tensor, s: float) -> torch.Tensor:
    # Ensure contiguous memory for optimal performance
    A_contig = A.contiguous()
    C = torch.empty_like(A_contig)

    n_elements = A_contig.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    scale_kernel[grid](
        A_contig,
        C,
        s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return C.view_as(A)


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scale(A, s)
