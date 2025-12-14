import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_pass1_kernel(
    x_ptr,
    out_ptr,
    block_sums_ptr,
    B,
    N,
    n_blocks,
    BLOCK_N: tl.constexpr,
):
    # pid along batch and block (along scan dimension)
    pid_b = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)

    block_start = pid_block * BLOCK_N
    offs = block_start + tl.arange(0, BLOCK_N)
    # mask for valid elements in this row & block
    mask = (pid_b < B) & (offs < N)

    row_start = pid_b * N

    # load block (invalid positions are 0 so they don't affect sums)
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)

    # inclusive cumsum within block
    x_scan = tl.cumsum(x, axis=0)

    # store local scan
    tl.store(out_ptr + row_start + offs, x_scan, mask=mask)

    # compute sum of this block (only valid elements contribute since invalid = 0)
    block_sum = tl.sum(x, axis=0)

    # store block sum (one per row & block)
    valid_block = (pid_b < B) & (pid_block < n_blocks)
    tl.store(block_sums_ptr + pid_b * n_blocks + pid_block, block_sum, mask=valid_block)


@triton.jit
def block_prefix_cumsum_kernel(
    block_sums_ptr,
    block_prefix_ptr,
    B,
    n_blocks,
    BLOCK_NBLOCKS: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    offs = tl.arange(0, BLOCK_NBLOCKS)
    mask = (pid_b < B) & (offs < n_blocks)

    row_offset = pid_b * n_blocks
    vals = tl.load(block_sums_ptr + row_offset + offs, mask=mask, other=0.0)

    vals_scan = tl.cumsum(vals, axis=0)

    tl.store(block_prefix_ptr + row_offset + offs, vals_scan, mask=mask)


@triton.jit
def cumsum_add_prefix_kernel(
    out_ptr,
    block_prefix_ptr,
    B,
    N,
    n_blocks,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)

    block_start = pid_block * BLOCK_N
    offs = block_start + tl.arange(0, BLOCK_N)
    mask = (pid_b < B) & (offs < N)

    row_start = pid_b * N

    # linear index into block_prefix for this (row, block)
    block_linear_idx = pid_b * n_blocks + pid_block

    # prefix to add = sum of all previous blocks (0 for block 0)
    prefix = tl.load(
        block_prefix_ptr + block_linear_idx - 1,
        mask=pid_block > 0,
        other=0.0,
    )

    vals = tl.load(out_ptr + row_start + offs, mask=mask, other=0.0)
    vals = vals + prefix
    tl.store(out_ptr + row_start + offs, vals, mask=mask)


def triton_inclusive_cumsum_lastdim_2d(x_2d: torch.Tensor) -> torch.Tensor:
    """
    Inclusive cumsum along the last dimension for a 2D contiguous tensor.
    x_2d: (B, N), CUDA tensor.
    Returns tensor of same shape with cumsum along dim=1.
    """
    assert x_2d.is_cuda
    assert x_2d.dim() == 2
    B, N = x_2d.shape

    if N == 0:
        return x_2d.clone()

    BLOCK_N = 1024  # power of 2
    n_blocks = triton.cdiv(N, BLOCK_N)

    out = torch.empty_like(x_2d)
    block_sums = torch.empty((B, n_blocks), device=x_2d.device, dtype=x_2d.dtype)
    block_prefix = torch.empty_like(block_sums)

    # Pass 1: per-block scan + per-block sums
    grid1 = (B, n_blocks)
    cumsum_pass1_kernel[grid1](
        x_2d,
        out,
        block_sums,
        B,
        N,
        n_blocks,
        BLOCK_N=BLOCK_N,
    )

    # Pass 2: prefix sum over block_sums along block dimension
    if n_blocks <= 1024:
        BLOCK_NBLOCKS = 1024  # power of 2
        grid2 = (B,)
        block_prefix_cumsum_kernel[grid2](
            block_sums,
            block_prefix,
            B,
            n_blocks,
            BLOCK_NBLOCKS=BLOCK_NBLOCKS,
        )
    else:
        # Fallback to PyTorch for very long sequences of blocks
        block_prefix.copy_(torch.cumsum(block_sums, dim=1))

    # Pass 3: add prefix to each block's local scan
    grid3 = (B, n_blocks)
    cumsum_add_prefix_kernel[grid3](
        out,
        block_prefix,
        B,
        N,
        n_blocks,
        BLOCK_N=BLOCK_N,
    )

    return out


def triton_inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Inclusive cumsum along an arbitrary dimension using Triton.
    """
    assert x.is_cuda
    ndim = x.ndim
    if ndim == 0:
        return x.clone()

    dim = dim % ndim

    # Move target dim to the last, then flatten outer dims into batch
    perm = list(range(ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x_perm = x.permute(*perm).contiguous()

    outer_shape = x_perm.shape[:-1]
    N = x_perm.shape[-1]
    B = int(torch.prod(torch.tensor(outer_shape))) if outer_shape else 1

    x_2d = x_perm.reshape(B, N)
    out_2d = triton_inclusive_cumsum_lastdim_2d(x_2d)
    out_perm = out_2d.reshape(*outer_shape, N).permute(*perm)

    return out_perm


def triton_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Reverse cumsum along the given dimension, implemented via
    Triton inclusive cumsum plus PyTorch flip.
    """
    dim = dim % x.ndim
    x_flipped = x.flip(dim)
    inclusive = triton_inclusive_cumsum(x_flipped, dim)
    return inclusive.flip(dim)


class ModelNew(nn.Module):
    """
    A Triton-accelerated model that performs a reverse cumulative sum operation
    along a specified dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)
