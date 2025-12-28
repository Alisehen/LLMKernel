import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tanh_kernel(
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

    # Use fast hardware/libdevice tanh approximation instead of explicit exp-based formula
    y = tl.math.tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    # Fallback to torch.tanh on CPU tensors
    if not x.is_cuda:
        return torch.tanh(x)

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()

    BLOCK_SIZE = 256  # power-of-2 as required
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    tanh_kernel[grid](
        x_contig,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Triton-accelerated Tanh activation model.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)
