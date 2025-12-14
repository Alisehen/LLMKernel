import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_pass1_kernel(
    x_ptr,              # *fptr: [B, N]
    out_ptr,            # *fptr: [B, N]
    block_sums_ptr,     # *fptr: [B, NB]
    N,                  # int: length of scan dimension
    stride_x_row,       # int: stride between rows in x/out
    stride_bs_row,      # int: stride between rows in block_sums
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_blk = tl.program_id(axis=1)

    block_start = pid_blk * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask for valid elements in this block
    mask = offsets < N

    row_offset_x = pid_row * stride_x_row

    x = tl.load(x_ptr + row_offset_x + offsets, mask=mask, other=0.0)

    # inclusive scan within the block
    cumsum = tl.cumsum(x, axis=0)

    # store per-element scanned values
    tl.store(out_ptr + row_offset_x + offsets, cumsum, mask=mask)

    # block sum is just sum of the block's elements
    block_sum = tl.sum(x, axis=0)

    # store per-block sums: [B, NB]
    bs_index = pid_row * stride_bs_row + pid_blk
    tl.store(block_sums_ptr + bs_index, block_sum)


@triton.jit
def _cumsum_add_prefix_kernel(
    out_ptr,            # *fptr: [B, N] (already has per-block scans)
    prefix_ptr,         # *fptr: [B, NB] (exclusive prefix of block sums)
    N,                  # int: length of scan dimension
    stride_out_row,     # int: stride between rows in out
    stride_prefix_row,  # int: stride between rows in prefix
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_blk = tl.program_id(axis=1)

    block_start = pid_blk * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    row_offset_out = pid_row * stride_out_row

    # scalar prefix value for this (row, block)
    prefix_index = pid_row * stride_prefix_row + pid_blk
    prefix_val = tl.load(prefix_ptr + prefix_index)

    vals = tl.load(out_ptr + row_offset_out + offsets, mask=mask, other=0.0)
    vals = vals + prefix_val
    tl.store(out_ptr + row_offset_out + offsets, vals, mask=mask)


def _triton_inclusive_cumsum_lastdim_2d(x_2d: torch.Tensor) -> torch.Tensor:
    """
    Inclusive cumsum along the last dimension of a 2D tensor using a multi-pass
    block-wise scan. x_2d is expected to be contiguous with shape [B, N].
    """
    assert x_2d.dim() == 2
    B, N = x_2d.shape
    if N == 0:
        return x_2d.clone()

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = 1024  # power-of-2 as required
    NB = triton.cdiv(N, BLOCK_SIZE)

    device = x_2d.device
    dtype = x_2d.dtype

    # Per-row, per-block sums: [B, NB]
    block_sums = torch.empty((B, NB), device=device, dtype=dtype)

    stride_x_row = x_2d.stride(0)
    stride_bs_row = block_sums.stride(0)

    # Pass 1: within-block scan and block sums
    grid1 = lambda meta: (B, triton.cdiv(N, meta['BLOCK_SIZE']))
    _cumsum_pass1_kernel[grid1](
        x_2d,
        out,
        block_sums,
        N,
        stride_x_row,
        stride_bs_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2 (PyTorch): prefix sums of block sums to get exclusive prefixes
    if NB > 1:
        inclusive = torch.cumsum(block_sums, dim=1)
        block_prefix = inclusive - block_sums   # exclusive prefix
    else:
        block_prefix = torch.zeros_like(block_sums)

    # Pass 3: add per-block prefix to each element
    stride_out_row = out.stride(0)
    stride_prefix_row = block_prefix.stride(0)

    grid3 = lambda meta: (B, triton.cdiv(N, meta['BLOCK_SIZE']))
    _cumsum_add_prefix_kernel[grid3](
        out,
        block_prefix,
        N,
        stride_out_row,
        stride_prefix_row,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def triton_inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Inclusive cumsum along arbitrary dimension `dim` using Triton kernels.
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
    Exclusive cumsum along dimension `dim`:
    out[i] = sum_{k < i} x[k]

    Implemented via inclusive Triton cumsum plus PyTorch shifting, per
    the required pattern.
    """
    inclusive = triton_inclusive_cumsum(x, dim)

    # Exclusive: shift right and pad a leading zero slice
    zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
    exclusive = torch.cat(
        [zeros, inclusive.narrow(dim, 0, x.size(dim) - 1)],
        dim=dim,
    )
    return exclusive


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)
