import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Numerically stable sigmoid implementation:
    # sigmoid(x) = 1 / (1 + exp(-x))          for x >= 0
    # sigmoid(x) = exp(x) / (1 + exp(x))      for x < 0
    zero = 0.0
    is_pos = x >= zero

    z_pos = tl.exp(-x)
    out_pos = 1.0 / (1.0 + z_pos)

    z_neg = tl.exp(x)
    out_neg = z_neg / (1.0 + z_neg)

    y = tl.where(is_pos, out_pos, out_neg)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Sigmoid activation using a Triton kernel.
    Falls back to torch.sigmoid for non-CUDA tensors.
    """
    if not x.is_cuda:
        return torch.sigmoid(x)

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    sigmoid_kernel[grid](x_contig, y, n_elements)

    # Preserve original non-contiguous layout if needed
    if x.data_ptr() != x_contig.data_ptr():
        return y.view_as(x)
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a Sigmoid activation using a Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)
