# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Nearby configs, keeping BLOCK_SIZE/grid fixed and including original
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),  # was stages=1
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),  # original config
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def _cumsum_rowwise_kernel(
    x_ptr,          # *fptr: [B, N]
    out_ptr,        # *fptr: [B, N]
    N,              # int32: length of scan dimension
    stride_x_row,   # int32: stride between rows in x
    stride_out_row, # int32: stride between rows in out
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass inclusive cumsum along the last dimension for a 2D tensor [B, N].

    Each program handles one row (batch index), and iterates over the row in
    BLOCK_SIZE chunks, carrying a running sum across chunks. This minimizes
    global memory traffic: each element is read once and written once.
    """
    pid_row = tl.program_id(axis=0)

    row_x_ptr = x_ptr + pid_row * stride_x_row
    row_out_ptr = out_ptr + pid_row * stride_out_row

    offsets = tl.arange(0, BLOCK_SIZE)
    start = 0
    # running prefix sum for this row; keep as float for numerical stability
    carry = 0.0

    # Iterate over the row in BLOCK_SIZE-sized chunks
    while start < N:
        idx = start + offsets
        mask = idx < N

        x = tl.load(row_x_ptr + idx, mask=mask, other=0)
        # In-block inclusive scan, then add running prefix
        cumsum = tl.cumsum(x, axis=0) + carry
        tl.store(row_out_ptr + idx, cumsum, mask=mask)

        # Update running prefix with this block's total sum
        carry += tl.sum(x, axis=0)

        start += BLOCK_SIZE


def _triton_inclusive_cumsum_lastdim_2d(x_2d: torch.Tensor) -> torch.Tensor:
    """
    Inclusive cumsum along the last dimension of a 2D tensor using a single-pass
    Triton kernel. x_2d is expected to be contiguous with shape [B, N].
    """
    assert x_2d.dim() == 2
    B, N = x_2d.shape
    if N == 0:
        return x_2d.clone()

    out = torch.empty_like(x_2d)

    stride_x_row = x_2d.stride(0)
    stride_out_row = out.stride(0)

    # One program per row; kernel loops over N internally
    grid = (B,)
    _cumsum_rowwise_kernel[grid](
        x_2d,
        out,
        N,
        stride_x_row,
        stride_out_row,
    )

    return out


def triton_inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Inclusive cumsum along arbitrary dimension `dim` using the optimized Triton kernel.
    """
    if dim < 0:
        dim = x.dim() + dim

    # Move target dim to last, make contiguous, and flatten leading dims
    if dim != x.dim() - 1:
        x_moved = x.movedim(dim, -1)
    else:
        x_moved = x

    x_contig = x_moved.contiguous()
    last_dim = x_contig.shape[-1]
    B = x_contig.numel() // last_dim
    x_2d = x_contig.view(B, last_dim)

    y_2d = _triton_inclusive_cumsum_lastdim_2d(x_2d)
    y_contig = y_2d.view(x_contig.shape)

    # Move dimension back to original position
    if dim != x.dim() - 1:
        y = y_contig.movedim(-1, dim)
    else:
        y = y_contig

    return y


def triton_exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Exclusive cumsum matching the provided PyTorch reference:

        exclusive_cumsum = torch.cat(
            (torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x),
            dim=dim,
        )[:-1]
        return torch.cumsum(exclusive_cumsum, dim=dim)

    Only the final torch.cumsum is replaced with the Triton implementation.
    """
    if dim < 0:
        dim = x.dim() + dim

    zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
    # Note: `[:-1]` slices along the first dimension, exactly as in the reference.
    exclusive_input = torch.cat((zeros, x), dim=dim)[:-1]
    return triton_inclusive_cumsum(exclusive_input, dim)


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element),
    matching the semantics of the given PyTorch reference model.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)
