import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_keepdim_kernel(
    x_ptr,
    out_ptr,
    reduce_size,
    inner_size,
    total_slices,
    BLOCK_SLICE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_slice = tl.program_id(axis=0)
    pid_reduce = tl.program_id(axis=1)

    slice_offsets = pid_slice * BLOCK_SLICE + tl.arange(0, BLOCK_SLICE)
    reduce_offsets = pid_reduce * BLOCK_R + tl.arange(0, BLOCK_R)

    slice_mask = slice_offsets < total_slices
    reduce_mask = reduce_offsets < reduce_size

    slice_offsets_i64 = slice_offsets.to(tl.int64)
    reduce_offsets_i64 = reduce_offsets.to(tl.int64)
    inner_i64 = tl.full((1,), inner_size, tl.int64)[0]
    reduce_i64 = tl.full((1,), reduce_size, tl.int64)[0]

    outer_idx = slice_offsets_i64 // inner_i64
    inner_idx = slice_offsets_i64 - outer_idx * inner_i64
    base = outer_idx * reduce_i64 * inner_i64 + inner_idx

    ptrs = base[:, None] + reduce_offsets_i64[None, :] * inner_i64
    mask = slice_mask[:, None] & reduce_mask[None, :]

    vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    partial = tl.sum(vals, axis=1)

    tl.atomic_add(out_ptr + slice_offsets_i64, partial, mask=slice_mask)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += x.ndim
    assert 0 <= dim < x.ndim, "Invalid reduction dimension"

    x = x.contiguous()
    reduce_size = x.shape[dim]
    out_shape = list(x.shape)
    out_shape[dim] = 1

    if reduce_size == 0:
        return torch.zeros(out_shape, device=x.device, dtype=x.dtype)

    inner_size = 1
    for i in range(dim + 1, x.ndim):
        inner_size *= x.shape[i]
    outer_size = x.numel() // (reduce_size * inner_size)
    total_slices = outer_size * inner_size

    if total_slices == 0:
        return torch.zeros(out_shape, device=x.device, dtype=x.dtype)

    out_accum = torch.zeros(out_shape, device=x.device, dtype=torch.float32)

    BLOCK_SLICE = 64
    BLOCK_R = 128

    grid = (
        triton.cdiv(total_slices, BLOCK_SLICE),
        triton.cdiv(reduce_size, BLOCK_R),
    )

    sum_reduce_keepdim_kernel[grid](
        x,
        out_accum,
        reduce_size,
        inner_size,
        total_slices,
        BLOCK_SLICE=BLOCK_SLICE,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )

    return out_accum.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_keepdim(x, self.dim)
