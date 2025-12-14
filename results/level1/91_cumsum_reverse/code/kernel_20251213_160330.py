import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
    ],
    key=["N"],
)
@triton.jit
def cumsum_lastdim_1pass_kernel(
    x_ptr,
    out_ptr,
    B,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    Single-pass inclusive cumsum along the last dimension for a 2D tensor (B, N).

    Each Triton program handles one row and iterates over N in tiles of size BLOCK_N.
    """
    pid_b = tl.program_id(axis=0)
    if pid_b >= B:
        return

    row_start = pid_b * N
    offs = tl.arange(0, BLOCK_N)

    # Running prefix for this row, scalar in same dtype as x
    prefix = 0

    # Tile over the N dimension
    for start in range(0, N, BLOCK_N):
        idx = start + offs
        mask = idx < N

        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0)

        # In-place inclusive scan over the BLOCK_N elements of x using
        # a parallel Hillis–Steele prefix-sum (log2(BLOCK_N) steps).
        # We avoid out-of-bounds indexing by clamping the source indices.
        # Step 1
        y = x
        pos = tl.where(offs >= 1, offs - 1, 0)
        shifted = y[pos]
        shifted = tl.where(offs >= 1, shifted, 0)
        x = x + shifted

        if BLOCK_N >= 2:
            y = x
            pos = tl.where(offs >= 2, offs - 2, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 2, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 4:
            y = x
            pos = tl.where(offs >= 4, offs - 4, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 4, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 8:
            y = x
            pos = tl.where(offs >= 8, offs - 8, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 8, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 16:
            y = x
            pos = tl.where(offs >= 16, offs - 16, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 16, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 32:
            y = x
            pos = tl.where(offs >= 32, offs - 32, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 32, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 64:
            y = x
            pos = tl.where(offs >= 64, offs - 64, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 64, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 128:
            y = x
            pos = tl.where(offs >= 128, offs - 128, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 128, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 256:
            y = x
            pos = tl.where(offs >= 256, offs - 256, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 256, shifted, 0)
            x = x + shifted

        if BLOCK_N >= 512:
            y = x
            pos = tl.where(offs >= 512, offs - 512, 0)
            shifted = y[pos]
            shifted = tl.where(offs >= 512, shifted, 0)
            x = x + shifted

        # Add running prefix from previous tiles
        x = x + prefix

        tl.store(out_ptr + row_start + idx, x, mask=mask)

        # Update prefix with the last value of the tile.
        # For a partially full tile, masked lanes are loaded as 0,
        # so the last lane still holds the sum over all valid elements.
        prefix = x[BLOCK_N - 1]


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

    out = torch.empty_like(x_2d)

    # 1D grid over batch; each program processes one row and loops over N.
    grid = (B,)
    cumsum_lastdim_1pass_kernel[grid](
        x_2d,
        out,
        B,
        N,
    )

    return out


def triton_inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Inclusive cumsum along an arbitrary dimension using a Triton 1-pass kernel
    along the last dimension.
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
    if outer_shape:
        B = int(torch.tensor(outer_shape).prod().item())
    else:
        B = 1

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
    along a specified dimension, using an optimized 1-pass kernel.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_reverse_cumsum(x, self.dim)
