import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_tile_kernel(
    x_ptr,               # pointer to input [M, N]
    y_ptr,               # pointer to output [M, N]
    block_sums_ptr,      # pointer to [M, num_blocks] (tile sums)
    M,                   # number of rows
    N,                   # number of columns (cumsum dim)
    num_blocks,          # number of tiles per row = ceil_div(N, BLOCK_SIZE)
    stride_m,            # stride between rows in elements
    stride_n,            # stride between columns in elements (usually 1)
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row index
    pid_b = tl.program_id(axis=1)  # block index within the row

    # starting column for this tile
    col_start = pid_b * BLOCK_SIZE

    # local indices [0, BLOCK_SIZE)
    idx = tl.arange(0, BLOCK_SIZE)
    # global column offsets for this tile
    offs = col_start + idx

    # mask for elements within the row length N
    mask = offs < N

    # base pointers for this row
    x_row_ptr = x_ptr + pid_m * stride_m
    y_row_ptr = y_ptr + pid_m * stride_m

    # load this tile, zero-padding out-of-range columns
    vals = tl.load(x_row_ptr + offs * stride_n, mask=mask, other=0.0)

    # compute tile sum BEFORE modifying vals (sum of original values)
    tile_sum = tl.sum(vals, axis=0)

    # write initial values to y (stage 0 of scan)
    tl.store(y_row_ptr + offs * stride_n, vals, mask=mask)

    # Hillis-Steele inclusive scan within the tile using global memory as scratch
    # BLOCK_SIZE is 256, so shifts up to 128
    for shift in (1, 2, 4, 8, 16, 32, 64, 128):
        src_idx = idx - shift
        # avoid negative indices in pointer arithmetic
        src_idx = tl.where(idx >= shift, src_idx, 0)
        prev_mask = mask & (idx >= shift)
        prev = tl.load(
            y_row_ptr + (col_start + src_idx) * stride_n,
            mask=prev_mask,
            other=0.0,
        )
        vals = vals + prev
        tl.store(y_row_ptr + offs * stride_n, vals, mask=mask)

    # store tile sum into [M, num_blocks]
    bs_offset = pid_m * num_blocks + pid_b
    tl.store(block_sums_ptr + bs_offset, tile_sum)


@triton.jit
def _add_block_offsets_kernel(
    y_ptr,                # pointer to output [M, N]
    block_offsets_ptr,    # pointer to [M, num_blocks] (exclusive prefix of tile sums)
    M,
    N,
    num_blocks,
    stride_m,
    stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row index
    pid_b = tl.program_id(axis=1)  # block index within row

    col_start = pid_b * BLOCK_SIZE
    idx = tl.arange(0, BLOCK_SIZE)
    offs = col_start + idx
    mask = offs < N

    # base pointer for this row
    y_row_ptr = y_ptr + pid_m * stride_m

    # scalar block offset for this tile
    bs_offset = pid_m * num_blocks + pid_b
    block_off = tl.load(block_offsets_ptr + bs_offset)

    # load current prefix sums, add block offset, and store back
    vals = tl.load(y_row_ptr + offs * stride_n, mask=mask, other=0.0)
    vals = vals + block_off
    tl.store(y_row_ptr + offs * stride_n, vals, mask=mask)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Masked cumulative sum using Triton, equivalent to torch.cumsum(x * mask, dim=dim).
    Assumes x and mask are on CUDA device.
    """
    assert x.device.type == "cuda", "Input tensor must be on CUDA for Triton kernels."
    assert mask.device == x.device, "Mask must be on the same device as input."
    assert x.shape == mask.shape, "x and mask must have the same shape."

    # Apply mask (boolean -> same dtype as x)
    x_masked = x * mask.to(x.dtype)

    if x_masked.numel() == 0:
        return x_masked

    dim = dim if dim >= 0 else x_masked.dim() + dim
    assert 0 <= dim < x_masked.dim()

    # Move cumsum dimension to the last axis and make contiguous
    if dim != x_masked.dim() - 1:
        x_perm = x_masked.transpose(dim, -1).contiguous()
        transposed = True
    else:
        x_perm = x_masked.contiguous()
        transposed = False

    M = x_perm.numel() // x_perm.size(-1)
    N = x_perm.size(-1)

    if N == 0:
        # Nothing to scan along cumsum dimension
        out_perm = x_perm
    else:
        x2d = x_perm.view(M, N)
        y2d = torch.empty_like(x2d)

        BLOCK_SIZE = 256  # power-of-two, as required
        num_blocks = triton.cdiv(N, BLOCK_SIZE)

        # First pass: per-tile prefix sums and tile sums
        block_sums = torch.empty(
            (M, num_blocks),
            device=x.device,
            dtype=x.dtype,
        )

        stride_m = x2d.stride(0)
        stride_n = x2d.stride(1)

        grid = (M, num_blocks)
        _cumsum_tile_kernel[grid](
            x2d,
            y2d,
            block_sums,
            M,
            N,
            num_blocks,
            stride_m,
            stride_n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Second pass: prefix sum over tiles per row on PyTorch (small, fast)
        # exclusive prefix: offset[b] = sum_{k < b} tile_sum[k]
        block_offsets = torch.cumsum(block_sums, dim=1) - block_sums

        # Third pass: add tile offsets to each element
        _add_block_offsets_kernel[grid](
            y2d,
            block_offsets,
            M,
            N,
            num_blocks,
            stride_m,
            stride_n,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        out_perm = y2d.view_as(x_perm)

    # Move back to original dimension order
    if transposed:
        out = out_perm.transpose(-1, dim)
    else:
        out = out_perm

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs masked cumulative sum along a given dimension:
        output = torch.cumsum(x * mask, dim=dim)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        return triton_masked_cumsum(x, mask, self.dim)
