# <complete ModelNew code with optimized Triton kernels>
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
    pid_slice = tl.program_id(0)
    pid_reduce = tl.program_id(1)

    slice_idx = pid_slice * BLOCK_SLICE + tl.arange(0, BLOCK_SLICE)
    reduce_idx = pid_reduce * BLOCK_R + tl.arange(0, BLOCK_R)

    slice_mask = slice_idx < total_slices
    reduce_mask = reduce_idx < reduce_size

    slice_idx64 = slice_idx.to(tl.int64)
    reduce_idx64 = reduce_idx.to(tl.int64)

    outer_idx = slice_idx64 // inner_size
    inner_idx = slice_idx64 - outer_idx * inner_size
    base = outer_idx * reduce_size * inner_size + inner_idx
    ptrs = base[:, None] + reduce_idx64[None, :] * inner_size

    mask = slice_mask[:, None] & reduce_mask[None, :]
    vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=1)

    tl.atomic_add(out_ptr + slice_idx64, acc, mask=slice_mask)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += x.ndim
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

    if inner_size >= 256:
        BLOCK_SLICE = 256
    elif inner_size >= 64:
        BLOCK_SLICE = 128
    else:
        BLOCK_SLICE = 64

    if reduce_size >= 512:
        BLOCK_R = 256
    elif reduce_size >= 128:
        BLOCK_R = 128
    else:
        BLOCK_R = 64

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
        num_warps=8 if BLOCK_R >= 256 else 4,
        num_stages=2,
    )

    return out_accum.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_keepdim(x, self.dim)
