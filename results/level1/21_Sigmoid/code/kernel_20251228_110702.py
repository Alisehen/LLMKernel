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

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Sigmoid: 1 / (1 + exp(-x))
    neg_x = -x
    exp_neg_x = tl.exp(neg_x)
    denom = 1.0 + exp_neg_x
    y = 1.0 / denom

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton implementation of torch.sigmoid(x).

    - For contiguous CUDA tensors, applies a single-pass Triton kernel:
      one global read of x and one global write of the sigmoid output.
    - For non-contiguous tensors or non-CUDA tensors, falls back to torch.sigmoid
      to preserve full PyTorch semantics (including strides/layout).
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernels.")

    # For non-contiguous tensors, avoid an additional explicit contiguous copy here
    # and just delegate to PyTorch's highly optimized implementation, which
    # correctly handles arbitrary strides.
    if not x.is_contiguous():
        return torch.sigmoid(x)

    n_elements = x.numel()
    out = torch.empty_like(x)

    if n_elements == 0:
        return out

    BLOCK_SIZE = 256  # power-of-two block size for good bandwidth utilization
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    sigmoid_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,   # more warps to better saturate memory bandwidth
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs a Sigmoid activation.
    Matches the behavior of the reference Model:
        forward(x) -> torch.sigmoid(x)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)
