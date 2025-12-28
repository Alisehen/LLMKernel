# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Aggressive large-chunk config for big tensors: fewer programs, higher ILP
        triton.Config({"BLOCK_SIZE": 256, "UNROLL": 8}, num_warps=8, num_stages=2),
        # More conservative config: good for medium sizes / slightly lower reg pressure
        triton.Config({"BLOCK_SIZE": 256, "UNROLL": 4}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def relu_kernel(
    x_ptr,          # *const T
    y_ptr,          # *T (may alias x_ptr for in-place)
    n_elements,     # int32
    BLOCK_SIZE: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """
    High-throughput, memory-bandwidth–saturating ReLU kernel.

    - Each program processes BLOCK_SIZE * UNROLL elements.
    - BLOCK_SIZE is a power of 2 (here 256) to match warp-friendly sizes.
    - UNROLL is compile-time (tl.static_range) to maximize ILP and reduce grid size.
    """
    pid = tl.program_id(axis=0)

    # Start index for this program's chunk
    block_start = pid * BLOCK_SIZE * UNROLL

    # Base offsets for one BLOCK
    base_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Unrolled loop: each iteration processes a contiguous BLOCK
    for u in tl.static_range(UNROLL):
        offsets = base_offsets + u * BLOCK_SIZE
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.maximum(x, 0.0)
        tl.store(y_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance ReLU using Triton.

    Fast path:
        - CUDA tensor
        - Contiguous layout
        - Computed in-place to minimize memory traffic

    Fallback:
        - Non-contiguous or non-CUDA tensors use torch.relu
    """
    # Fallback for non-CUDA tensors
    if not x.is_cuda:
        return torch.relu(x)

    # Fast path for contiguous CUDA tensors: in-place ReLU
    if x.is_contiguous():
        n_elements = x.numel()
        if n_elements == 0:
            return x  # nothing to do

        # Use views to get flat pointers; storage is shared with x
        x_flat = x.view(-1)
        y_flat = x_flat  # in-place: input and output share the same storage

        # Grid: 1D over flattened elements; each program handles BLOCK_SIZE * UNROLL
        def grid(meta):
            # Ensure grid size > 0 whenever n_elements > 0
            return (triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["UNROLL"]),)

        relu_kernel[grid](
            x_flat,
            y_flat,
            n_elements,
        )

        return x

    # Non-contiguous: rely on PyTorch's highly optimized implementation
    return torch.relu(x)


class ModelNew(nn.Module):
    """
    Model using a high-performance Triton ReLU kernel.

    Note:
        For best performance, pass CUDA, contiguous tensors.
        Non-contiguous or CPU tensors fall back to torch.relu.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
