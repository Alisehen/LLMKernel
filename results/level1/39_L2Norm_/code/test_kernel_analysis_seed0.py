# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l2norm_sumsq_kernel(
    x_ptr,             # *f32 / *f16, [B, D]
    norms_sqr_ptr,     # *f32 / *f16, [B]
    D,                 # int32, number of columns
    stride_x_batch,    # int32
    stride_x_dim,      # int32
    CHUNK_SIZE: tl.constexpr,        # columns per tile, power-of-2
    TILES_PER_PROGRAM: tl.constexpr, # tiles per program along dim
):
    pid_batch = tl.program_id(axis=0)
    pid_tile_group = tl.program_id(axis=1)

    row_x_ptr = x_ptr + pid_batch * stride_x_batch

    # number of tiles along D dimension
    T = (D + CHUNK_SIZE - 1) // CHUNK_SIZE

    acc = tl.zeros((), dtype=tl.float32)

    # process up to TILES_PER_PROGRAM tiles per program
    for t in range(TILES_PER_PROGRAM):
        tile_id = pid_tile_group * TILES_PER_PROGRAM + t
        tile_start = tile_id * CHUNK_SIZE

        # skip completely out-of-range tiles
        if tile_id < T:
            offs = tile_start + tl.arange(0, CHUNK_SIZE)
            mask = offs < D

            x = tl.load(row_x_ptr + offs * stride_x_dim, mask=mask, other=0.0)
            # accumulate in float32 for numerical stability
            x_f32 = x.to(tl.float32)
            acc += tl.sum(x_f32 * x_f32, axis=0)

    # atomic add to the row's sum-of-squares
    tl.atomic_add(norms_sqr_ptr + pid_batch, acc)


@triton.jit
def l2norm_sqrt_kernel(
    norms_sqr_ptr,  # *f32 / *f16, [B]
    norms_ptr,      # *f32 / *f16, [B]
    B,              # int32, number of rows
):
    pid = tl.program_id(axis=0)

    # grid is exactly B, so no extra bounds check needed
    val = tl.load(norms_sqr_ptr + pid)
    val_f32 = val.to(tl.float32)
    norm = tl.sqrt(val_f32)
    tl.store(norms_ptr + pid, norm.to(val.dtype))


@triton.jit
def l2norm_normalize_kernel(
    x_ptr,            # *f32 / *f16, [B, D]
    norms_ptr,        # *f32 / *f16, [B]
    y_ptr,            # *f32 / *f16, [B, D]
    D,                # int32, number of columns
    stride_x_batch,   # int32
    stride_x_dim,     # int32
    stride_y_batch,   # int32
    stride_y_dim,     # int32
    CHUNK_SIZE: tl.constexpr,        # columns per program tile
):
    pid_batch = tl.program_id(axis=0)
    pid_tile = tl.program_id(axis=1)

    row_x_ptr = x_ptr + pid_batch * stride_x_batch
    row_y_ptr = y_ptr + pid_batch * stride_y_batch

    col_start = pid_tile * CHUNK_SIZE
    offs = col_start + tl.arange(0, CHUNK_SIZE)
    mask = offs < D

    x = tl.load(row_x_ptr + offs * stride_x_dim, mask=mask, other=0.0)
    norm = tl.load(norms_ptr + pid_batch)

    # perform division in higher precision then cast back
    x_f32 = x.to(tl.float32)
    norm_f32 = norm.to(tl.float32)
    y = (x_f32 / norm_f32).to(x.dtype)

    tl.store(row_y_ptr + offs * stride_y_dim, y, mask=mask)


def triton_l2norm(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize each row of a 2D tensor [B, D] using Triton kernels.
    The output has the same dtype and device as the input.
    """
    assert x.ndim == 2, "Input must be 2D tensor [batch, dim]"
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernels.")

    B, D = x.shape
    if B == 0 or D == 0:
        # Degenerate case: nothing to normalize
        return x.clone()

    device = x.device
    dtype = x.dtype

    CHUNK_SIZE = 256          # power-of-2, good for performance
    TILES_PER_PROGRAM = 4     # how many tiles each program handles in sumsq kernel

    # Allocate intermediate buffers: one scalar per row
    norms_sqr = torch.zeros((B,), device=device, dtype=torch.float32)
    norms = torch.empty((B,), device=device, dtype=torch.float32)

    # Output tensor, same layout as input
    y = torch.empty_like(x)

    stride_x_batch, stride_x_dim = x.stride()
    stride_y_batch, stride_y_dim = y.stride()

    # 1) Compute sum of squares along dim=1 using atomics
    grid_sumsq = lambda meta: (
        B,
        max(1, triton.cdiv(D, meta["CHUNK_SIZE"] * meta["TILES_PER_PROGRAM"])),
    )
    l2norm_sumsq_kernel[grid_sumsq](
        x,
        norms_sqr,
        D,
        stride_x_batch,
        stride_x_dim,
        CHUNK_SIZE=CHUNK_SIZE,
        TILES_PER_PROGRAM=TILES_PER_PROGRAM,
        num_warps=4,
        num_stages=2,
    )

    # 2) Take square root to obtain L2 norms
    grid_sqrt = lambda meta: (B,)
    l2norm_sqrt_kernel[grid_sqrt](
        norms_sqr,
        norms,
        B,
        num_warps=1,
        num_stages=1,
    )

    # 3) Normalize x by its L2 norm along dim=1
    grid_norm = lambda meta: (
        B,
        max(1, triton.cdiv(D, meta["CHUNK_SIZE"])),
    )
    l2norm_normalize_kernel[grid_norm](
        x,
        norms,
        y,
        D,
        stride_x_batch,
        stride_x_dim,
        stride_y_batch,
        stride_y_dim,
        CHUNK_SIZE=CHUNK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Cast back to original dtype if needed
    if dtype != torch.float32:
        return y.to(dtype)
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs L2 normalization along dim=1.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2norm(x)
