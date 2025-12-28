import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def elu_kernel(
    x_ptr,
    y_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    pos = x
    neg = alpha * (tl.exp(x) - 1.0)
    y = tl.where(x > 0.0, pos, neg)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_elu(x: torch.Tensor, alpha: float) -> torch.Tensor:
    # Fallback for non-CUDA tensors
    if not x.is_cuda:
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1.0))

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    x_flat = x_contig.view(-1)
    y_flat = y.view(-1)

    n_elements = x_flat.numel()
    BLOCK_SIZE = 256

    grid = lambda meta: (
        triton.cdiv(max(1, n_elements), meta["BLOCK_SIZE"]),
    )

    elu_kernel[grid](
        x_flat,
        y_flat,
        alpha,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs an ELU activation.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu(x, self.alpha)
