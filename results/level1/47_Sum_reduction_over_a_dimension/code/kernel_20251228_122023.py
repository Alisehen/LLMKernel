import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def sum_reduce_dim_kernel(
    x_ptr,       # *f32 / *f16 / *bf16
    out_ptr,     # *f32
    M,           # number of outer batches
    R,           # reduction length
    N,           # number of inner elements
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Program IDs: along N and M
    pid_n = tl.program_id(axis=0)  # tile along N
    pid_m = tl.program_id(axis=1)  # index along M

    m = pid_m
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for valid indices
    mask_n = n_offsets < N
    mask_m = m < M

    # Base offset for this m
    base_m = m * R * N

    # Accumulator for reduction result for each n in the tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    r_start = 0
    while r_start < R:
        # Tile over the reduction dimension: [BLOCK_R]
        r_offsets = r_start + tl.arange(0, BLOCK_R)

        # 2D tile indices: [BLOCK_R, BLOCK_N]
        r_broadcast = r_offsets[:, None]
        n_broadcast = n_offsets[None, :]

        # Linear indices into x[m, r, n] with layout [M, R, N]
        idx = base_m + r_broadcast * N + n_broadcast

        # Validity mask for this tile
        mask_r = r_offsets < R
        mask_tile = (mask_r[:, None]) & (mask_n[None, :]) & mask_m

        # Load tile, promote to fp32, and reduce along R axis (0)
        x_tile = tl.load(x_ptr + idx, mask=mask_tile, other=0.0)
        x_tile = x_tile.to(tl.float32)
        tile_sum = tl.sum(x_tile, axis=0)  # [BLOCK_N]

        acc += tile_sum
        r_start += BLOCK_R

    # Store result at out[m, 0, n]
    out_base = m * N + n_offsets  # out layout is [M, 1, N]
    tl.store(out_ptr + out_base, acc, mask=mask_n & mask_m)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum reduction over a given dim with keepdim=True using Triton."""
    # Fallback for non-CUDA or unsupported dtypes
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return torch.sum(x, dim=dim, keepdim=True)

    ndim = x.ndim
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim

    orig_shape = list(x.shape)

    # Flatten to [M, R, N] where we reduce over R (axis=1)
    if dim == 0:
        M = 1
        R = orig_shape[0]
        N = int(torch.prod(torch.tensor(orig_shape[1:], device='cpu'))) if ndim > 1 else 1
    elif dim == ndim - 1:
        M = int(torch.prod(torch.tensor(orig_shape[:-1], device='cpu'))) if ndim > 1 else 1
        R = orig_shape[-1]
        N = 1
    else:
        M = int(torch.prod(torch.tensor(orig_shape[:dim], device='cpu')))
        R = orig_shape[dim]
        N = int(torch.prod(torch.tensor(orig_shape[dim + 1:], device='cpu'))) if dim + 1 < ndim else 1

    # Ensure contiguous before reshape for predictable layout
    x_contig = x.contiguous()
    x_reshaped = x_contig.reshape(M, R, N)

    # Accumulate in fp32 for numerical stability
    out_reshaped = torch.empty((M, 1, N), device=x.device, dtype=torch.float32)

    # Tunable block sizes (power-of-two as required)
    BLOCK_N = 128
    BLOCK_R = 128

    def grid(meta):
        return (
            max(1, triton.cdiv(N, meta['BLOCK_N'])),  # along N
            max(1, M),                                # along M
        )

    sum_reduce_dim_kernel[grid](
        x_reshaped,
        out_reshaped,
        M,
        R,
        N,
        BLOCK_N=BLOCK_N,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=2,
    )

    # Cast back to original dtype if needed
    if x.dtype != torch.float32:
        out_reshaped = out_reshaped.to(x.dtype)

    # Reshape back to original shape with reduced dim kept as size 1
    out_shape = orig_shape.copy()
    out_shape[dim] = 1
    return out_reshaped.reshape(out_shape)


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_keepdim(x, self.dim)
