# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Sigmoid: 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    denom = 1.0 + exp_neg_x
    y = 1.0 / denom

    tl.store(out_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton implementation of torch.sigmoid(x).
    Works on any contiguous CUDA tensor (any shape, dtype float32/float16/bfloat16).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernels.")
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = out.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 256  # power-of-two, as required
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    sigmoid_kernel[grid](
        x_contig,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs a Sigmoid activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)
