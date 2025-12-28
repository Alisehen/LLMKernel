import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduce_lastdim_kernel(
    x_ptr,          # *f32 / *f16 / ...
    out_ptr,        # *same as x
    M,              # number of rows
    N,              # reduction length (last dimension)
    stride_xm,      # stride over rows in x
    stride_xn,      # stride over columns in x (should be 1 for contiguous last dim)
    NUM_CHUNKS: tl.constexpr,  # number of BLOCK_N-sized chunks along N
    BLOCK_N: tl.constexpr,     # tile size along reduction dim (power of 2)
):
    pid = tl.program_id(axis=0)
    row_mask = pid < M

    # Base pointer for this row
    row_ptr = x_ptr + pid * stride_xm

    # Tile offsets reused for all chunks
    offs = tl.arange(0, BLOCK_N)

    # --- First chunk: initialize row_min ---
    col_start = 0
    cols = col_start + offs
    mask0 = row_mask & (cols < N)

    # Use +inf for invalid lanes so they don't affect min
    vals = tl.load(row_ptr + cols * stride_xn, mask=mask0, other=float("inf"))
    # min(x) = -max(-x)
    row_min = -tl.max(-vals, axis=0)

    # --- Remaining chunks ---
    # NUM_CHUNKS is constexpr so this loop is fully unrolled
    for chunk in range(1, NUM_CHUNKS):
        col_start = chunk * BLOCK_N
        cols = col_start + offs
        mask = row_mask & (cols < N)
        vals = tl.load(row_ptr + cols * stride_xn, mask=mask, other=float("inf"))
        tile_min = -tl.max(-vals, axis=0)
        row_min = tl.minimum(row_min, tile_min)

    tl.store(out_ptr + pid, row_min, mask=row_mask)


def triton_min_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute torch.min(x, dim=dim)[0] using a high-performance Triton kernel.
    Assumes CUDA tensor input.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    ndim = x.dim()
    if dim < 0:
        dim = ndim + dim
    assert 0 <= dim < ndim

    # Move reduction dimension to the last axis for contiguous reduction
    x_moved = x.transpose(dim, -1).contiguous()
    *prefix_shape, N = x_moved.shape
    M = x_moved.numel() // N

    # 2D view: (M, N) where N is the reduction dimension
    x_2d = x_moved.view(M, N)

    out_flat = torch.empty(M, device=x.device, dtype=x.dtype)

    BLOCK_N = 128  # power-of-two for good performance
    NUM_CHUNKS = triton.cdiv(N, BLOCK_N)

    # Ensure grid size > 0
    grid = lambda meta: (max(1, M),)

    min_reduce_lastdim_kernel[grid](
        x_2d,
        out_flat,
        M,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        NUM_CHUNKS=NUM_CHUNKS,
        BLOCK_N=BLOCK_N,
    )

    # Reshape back to all non-reduced dimensions (already in correct order)
    out = out_flat.view(*prefix_shape)
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs min reduction over a specific dimension.
    Returns the same result as torch.min(x, dim=dim)[0].
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_dim(x, self.dim)
