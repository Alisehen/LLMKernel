import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _cumsum_tile_kernel(
    x_ptr,               # *x_dtype, [M, N]
    mask_ptr,            # *bool,    [M, N]
    y_ptr,               # *x_dtype, [M, N]
    M,                   # number of rows
    N,                   # number of columns (cumsum dim)
    stride_m,            # row stride in elements
    stride_n,            # col stride in elements (usually 1)
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row index
    pid_b = tl.program_id(axis=1)  # block index within the row

    # starting column for this tile
    col_start = pid_b * BLOCK_SIZE

    # local indices [0, BLOCK_SIZE)
    idx = tl.arange(0, BLOCK_SIZE)
    offs = col_start + idx

    # mask for elements within the row length N
    in_bounds = offs < N

    # base pointers for this row
    x_row_ptr = x_ptr + pid_m * stride_m
    y_row_ptr = y_ptr + pid_m * stride_m
    mask_row_ptr = mask_ptr + pid_m * stride_m

    # load this tile of x and mask, zero-padding out-of-range columns
    x_vals = tl.load(x_row_ptr + offs * stride_n, mask=in_bounds, other=0.0)
    mask_vals = tl.load(mask_row_ptr + offs * stride_n, mask=in_bounds, other=0)

    # apply mask on the fly: only sum elements where mask is True
    vals = tl.where(mask_vals, x_vals, 0.0)

    # perform an inclusive scan within the tile in registers
    vals = tl.cumsum(vals, axis=0)

    # store back
    tl.store(y_row_ptr + offs * stride_n, vals, mask=in_bounds)


@triton.jit
def _add_block_offsets_kernel(
    y_ptr,                # *x_dtype, [M, N]
    block_offsets_ptr,    # *x_dtype, [M, num_blocks]
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
    in_bounds = offs < N

    # base pointer for this row
    y_row_ptr = y_ptr + pid_m * stride_m

    # scalar block offset for this tile
    bs_offset = pid_m * num_blocks + pid_b
    block_off = tl.load(block_offsets_ptr + bs_offset)

    # load current prefix sums, add block offset, and store back
    vals = tl.load(y_row_ptr + offs * stride_n, mask=in_bounds, other=0.0)
    vals = vals + block_off
    tl.store(y_row_ptr + offs * stride_n, vals, mask=in_bounds)


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Masked cumulative sum using Triton, equivalent to torch.cumsum(x * mask, dim=dim).
    Assumes x and mask are on CUDA device.
    """
    assert x.device.type == "cuda", "Input tensor must be on CUDA for Triton kernels."
    assert mask.device == x.device, "Mask must be on the same device as input."
    assert x.shape == mask.shape, "x and mask must have the same shape."
    assert mask.dtype == torch.bool, "Mask must be a boolean tensor."

    if x.numel() == 0:
        # Match torch.cumsum behavior on empty tensors
        return torch.empty_like(x)

    dim = dim if dim >= 0 else x.dim() + dim
    assert 0 <= dim < x.dim()

    # Move cumsum dimension to the last axis and make contiguous
    if dim != x.dim() - 1:
        x_perm = x.transpose(dim, -1).contiguous()
        mask_perm = mask.transpose(dim, -1).contiguous()
        transposed = True
    else:
        x_perm = x.contiguous()
        mask_perm = mask.contiguous()
        transposed = False

    M = x_perm.numel() // x_perm.size(-1)
    N = x_perm.size(-1)

    if N == 0:
        # Nothing to scan along cumsum dimension
        out_perm = torch.empty_like(x_perm)
    else:
        x2d = x_perm.view(M, N)
        mask2d = mask_perm.view(M, N)
        y2d = torch.empty_like(x2d)

        BLOCK_SIZE = 256  # power-of-two, as required
        num_blocks = triton.cdiv(N, BLOCK_SIZE)

        stride_m = x2d.stride(0)
        stride_n = x2d.stride(1)

        grid = (M, num_blocks)

        # First pass: per-tile prefix sums with mask fused into loads
        _cumsum_tile_kernel[grid](
            x2d,
            mask2d,
            y2d,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Compute tile sums from the first-pass outputs:
        # For each tile, the last in-bounds element of that tile in y2d
        # is the sum of masked x within the tile.
        #
        # tile b covers columns [b*BLOCK_SIZE, (b+1)*BLOCK_SIZE - 1] (clipped by N).
        # So last index = min(N, (b+1)*BLOCK_SIZE) - 1.
        idx_last = (
            torch.arange(1, num_blocks + 1, device=x2d.device, dtype=torch.int64)
            * BLOCK_SIZE
            - 1
        )
        idx_last = torch.clamp(idx_last, max=N - 1)
        idx_last_row = idx_last.unsqueeze(0).expand(M, -1)
        block_sums = y2d.gather(1, idx_last_row)

        # Second pass: prefix sum over tiles per row on PyTorch (small, fast)
        # exclusive prefix: offset[b] = sum_{k < b} tile_sum[k]
        if num_blocks == 1:
            block_offsets = torch.zeros_like(block_sums)
        else:
            zeros_col = torch.zeros_like(block_sums[:, :1])
            prefix_input = torch.cat([zeros_col, block_sums[:, :-1]], dim=1)
            block_offsets = torch.cumsum(prefix_input, dim=1)

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
            num_warps=4,
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
