# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Manually implemented math helpers for potential future use
# -----------------------------------------------------------------------------


def _tl_tanh(x):
    # Numerically stable tanh
    e2x = tl.exp(2 * x)
    return (e2x - 1) / (e2x + 1)


def _tl_sigmoid(x):
    return 1 / (1 + tl.exp(-x))


def _tl_gelu(x):
    # Approximate GELU used by PyTorch
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    return 0.5 * x * (1 + _tl_tanh(k0 * (x + k1 * x * x * x)))


def _tl_silu(x):
    return x * _tl_sigmoid(x)


def _tl_mish(x):
    # mish(x) = x * tanh(softplus(x))
    softplus = tl.log(1 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0)
    return x * _tl_tanh(softplus)


def _tl_softmax(x, axis: int = -1):
    # Simple softmax along given axis
    x_max = tl.max(x, axis=axis)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=axis)
    return exp_x / denom


# Attach to tl.* if not present (for compatibility)
if not hasattr(tl, "tanh"):
    tl.tanh = _tl_tanh
if not hasattr(tl, "sigmoid"):
    tl.sigmoid = _tl_sigmoid
if not hasattr(tl, "gelu"):
    tl.gelu = _tl_gelu
if not hasattr(tl, "silu"):
    tl.silu = _tl_silu
if not hasattr(tl, "mish"):
    tl.mish = _tl_mish
if not hasattr(tl, "softmax"):
    tl.softmax = _tl_softmax


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _cumsum_tile_kernel(
    x_ptr,               # *x_dtype, [M, N]
    mask_ptr,            # *bool,    [M, N]
    y_ptr,               # *x_dtype, [M, N]
    block_sums_ptr,      # *x_dtype, [M, num_blocks]
    M,                   # number of rows
    N,                   # number of columns (cumsum dim)
    stride_m,            # row stride for x/y in elements
    stride_n,            # col stride for x/y in elements (usually 1)
    stride_sums_m,       # row stride for block_sums in elements
    stride_sums_b,       # block (tile) stride for block_sums in elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    First pass:
      - compute masked inclusive cumsum within each BLOCK_SIZE tile
      - write tile-wise cumsums to y
      - write per-tile sums to block_sums (one scalar per row/tile)
    """
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

    # apply mask: only sum elements where mask is True
    masked_vals = tl.where(mask_vals, x_vals, 0.0)

    # perform an inclusive scan within the tile in registers
    vals = tl.cumsum(masked_vals, axis=0)

    # tile sum = sum of masked values in this tile
    tile_sum = tl.sum(masked_vals, axis=0)

    # store back per-element cumsums
    tl.store(y_row_ptr + offs * stride_n, vals, mask=in_bounds)

    # write per-tile sum: shape [M, num_blocks]
    sums_base_ptr = block_sums_ptr + pid_m * stride_sums_m + pid_b * stride_sums_b
    tl.store(sums_base_ptr, tile_sum)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def _add_block_offsets_kernel(
    y_ptr,                # *x_dtype, [M, N]
    block_offsets_ptr,    # *x_dtype, [M, num_blocks]
    M,
    N,
    num_blocks,
    stride_m,             # row stride for y
    stride_n,             # col stride for y
    stride_offs_m,        # row stride for block_offsets
    stride_offs_b,        # block (tile) stride for block_offsets
    BLOCK_SIZE: tl.constexpr,
):
    """
    Third pass:
      - add per-tile prefix offsets (exclusive scan over tile sums)
        to each element in y.
    """
    pid_m = tl.program_id(axis=0)  # row index
    pid_b = tl.program_id(axis=1)  # block index within row

    col_start = pid_b * BLOCK_SIZE
    idx = tl.arange(0, BLOCK_SIZE)
    offs = col_start + idx
    in_bounds = offs < N

    # base pointer for this row
    y_row_ptr = y_ptr + pid_m * stride_m

    # scalar block offset for this tile
    offs_ptr = block_offsets_ptr + pid_m * stride_offs_m + pid_b * stride_offs_b
    block_off = tl.load(offs_ptr)

    # load current prefix sums, add block offset, and store back
    vals = tl.load(y_row_ptr + offs * stride_n, mask=in_bounds, other=0.0)
    vals = vals + block_off
    tl.store(y_row_ptr + offs * stride_n, vals, mask=in_bounds)


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------


def triton_masked_cumsum(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Masked cumulative sum using Triton, equivalent to:
        torch.cumsum(x * mask, dim=dim)
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

        # Fixed, tuned tile size (power of 2)
        BLOCK_SIZE = 128
        num_blocks = triton.cdiv(N, BLOCK_SIZE)

        stride_m = x2d.stride(0)
        stride_n = x2d.stride(1)

        # Workspace for per-tile sums and offsets: [M, num_blocks]
        block_sums = torch.empty(
            (M, num_blocks), device=x2d.device, dtype=x2d.dtype
        )
        block_offsets = torch.empty_like(block_sums)

        stride_sums_m = block_sums.stride(0)
        stride_sums_b = block_sums.stride(1)

        # Grid is expressed as a function of meta to work with autotune
        grid = lambda META: (
            M,
            triton.cdiv(N, META["BLOCK_SIZE"]),
        )

        # First pass: per-tile prefix sums with mask fused into loads,
        # plus per-tile sums written to block_sums.
        _cumsum_tile_kernel[grid](
            x2d,
            mask2d,
            y2d,
            block_sums,
            M,
            N,
            stride_m,
            stride_n,
            stride_sums_m,
            stride_sums_b,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Second pass: prefix sum over tiles per row on PyTorch (small, fast)
        # exclusive prefix: offset[b] = sum_{k < b} tile_sum[k]
        if num_blocks == 1:
            block_offsets.zero_()
        else:
            zeros_col = torch.zeros_like(block_sums[:, :1])
            prefix_input = torch.cat([zeros_col, block_sums[:, :-1]], dim=1)
            block_offsets = torch.cumsum(prefix_input, dim=1)

        stride_offs_m = block_offsets.stride(0)
        stride_offs_b = block_offsets.stride(1)

        grid_offsets = (M, num_blocks)

        # Third pass: add tile offsets to each element
        _add_block_offsets_kernel[grid_offsets](
            y2d,
            block_offsets,
            M,
            N,
            num_blocks,
            stride_m,
            stride_n,
            stride_offs_m,
            stride_offs_b,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        out_perm = y2d.view_as(x_perm)

    # Move back to original dimension order
    if transposed:
        out = out_perm.transpose(-1, dim)
    else:
        out = out_perm

    return out


# -----------------------------------------------------------------------------
# nn.Module wrapper
# -----------------------------------------------------------------------------


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
