import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scalar_mul_kernel(
    a_ptr,          # *const float32
    out_ptr,        # *float32
    scalar,         # float32 scalar
    n_elements,     # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    out = a * scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_scalar_mul(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    A: (...,), any shape, CUDA tensor
    s: Python float or 0-dim tensor
    """
    A_contig = A.contiguous()
    out = torch.empty_like(A_contig)

    n_elements = A_contig.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    scalar_mul_kernel[grid](
        A_contig,
        out,
        float(s),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    using a high-performance Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_scalar_mul(A, s)
