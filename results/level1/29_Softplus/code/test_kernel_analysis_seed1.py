import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Numerically stable softplus:
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    abs_x = tl.abs(x)
    s = tl.maximum(x, 0.0)
    exp_term = tl.exp(-abs_x)
    out = s + tl.log(1.0 + exp_term)

    # Store result
    tl.store(y_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements == 0:
        return y

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    softplus_kernel[grid](
        x,
        y,
        n_elements,
    )
    return y


class ModelNew(nn.Module):
    """
    Softplus activation implemented with a high-performance Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)
