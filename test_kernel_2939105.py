import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    n_rows,
    L: tl.constexpr,           # sequence length along dim
    BLOCK_ROWS: tl.constexpr,  # number of rows processed per program (power of 2)
):
    pid = tl.program_id(axis=0)
    row_block_start = pid * BLOCK_ROWS

    # rows this program is responsible for
    row_idxs = row_block_start + tl.arange(0, BLOCK_ROWS)
    row_mask = row_idxs < n_rows

    # running sum per row
    running = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

    # iterate from last column to first (reverse cumsum)
    for col in range(L):
        col_rev = L - 1 - col

        x_offsets = row_idxs * stride_x + col_rev
        y_offsets = row_idxs * stride_y + col_rev

        x_vals = tl.load(x_ptr + x_offsets, mask=row_mask, other=0.0)
        x_vals = x_vals.to(tl.float32)

        running = running + x_vals

        tl.store(y_ptr + y_offsets, running, mask=row_mask)


def triton_reverse_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    orig_dtype = x.dtype
    if orig_dtype != torch.float32:
        x = x.to(torch.float32)

    ndim = x.ndim
    if dim < 0:
        dim += ndim

    # move target dim to last for contiguous 2D view
    if dim != ndim - 1:
        x = x.transpose(dim, -1).contiguous()
    else:
        x = x.contiguous()

    shape_last = x.shape
    L = shape_last[-1]
    n_rows = x.numel() // L

    x_2d = x.view(n_rows, L)
    y_2d = torch.empty_like(x_2d)

    stride_x = x_2d.stride(0)
    stride_y = y_2d.stride(0)

    BLOCK_ROWS = 256  # power-of-2, good occupancy

    grid = lambda meta: (triton.cdiv(n_rows, meta['BLOCK_ROWS']),)

    reverse_cumsum_kernel[grid](
        x_2d,
        y_2d,
        stride_x,
        stride_y,
        n_rows,
        L=L,
        BLOCK_ROWS=BLOCK_ROWS,
    )

    y = y_2d.view(shape_last)
    if dim != ndim - 1:
        y = y.transpose(dim, -1).contiguous()

    if orig_dtype != torch.float32:
        y = y.to(orig_dtype)
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized model computing reverse cumulative sum along a given dim:
        y = cumsum(x.flip(dim), dim=dim).flip(dim)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)
