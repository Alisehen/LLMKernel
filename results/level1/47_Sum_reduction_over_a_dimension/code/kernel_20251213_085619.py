# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SLICE': 64, 'BLOCK_R': 64, 'GROUP_R': 2, 'num_warps': 4, 'num_stages': 2}
        ),
        triton.Config(
            {'BLOCK_SLICE': 128, 'BLOCK_R': 64, 'GROUP_R': 4, 'num_warps': 8, 'num_stages': 2}
        ),
        triton.Config(
            {'BLOCK_SLICE': 64, 'BLOCK_R': 128, 'GROUP_R': 2, 'num_warps': 4, 'num_stages': 2}
        ),
        triton.Config(
            {'BLOCK_SLICE': 128, 'BLOCK_R': 128, 'GROUP_R': 2, 'num_warps': 8, 'num_stages': 2}
        ),
        triton.Config(
            {'BLOCK_SLICE': 256, 'BLOCK_R': 64, 'GROUP_R': 4, 'num_warps': 8, 'num_stages': 3}
        ),
    ],
    key=['inner_size', 'reduce_size'],
)
@triton.jit
def sum_reduce_keepdim_kernel(
    x_ptr,
    out_ptr,
    reduce_size,
    inner_size,
    stride_outer,
    stride_reduce,
    BLOCK_SLICE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP_R: tl.constexpr,
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    pid_group = tl.program_id(2)

    inner_offsets = pid_inner * BLOCK_SLICE + tl.arange(0, BLOCK_SLICE)
    inner_mask = inner_offsets < inner_size
    inner_offsets = inner_offsets.to(tl.int64)

    stride_outer_i64 = tl.full((), stride_outer, tl.int64)
    stride_reduce_i64 = tl.full((), stride_reduce, tl.int64)
    inner_size_i64 = tl.full((), inner_size, tl.int64)
    reduce_size_i64 = tl.full((), reduce_size, tl.int64)

    outer_offset = tl.cast(pid_outer, tl.int64) * stride_outer_i64
    base_ptrs = outer_offset + inner_offsets

    acc = tl.zeros((BLOCK_SLICE,), dtype=tl.float32)
    r_idx = tl.arange(0, BLOCK_R).to(tl.int64)
    group_span = BLOCK_R * GROUP_R
    group_base = tl.cast(pid_group, tl.int64) * group_span

    for g in tl.static_range(GROUP_R):
        reduce_offsets = group_base + g * BLOCK_R + r_idx
        mask_reduce = reduce_offsets < reduce_size_i64
        ptrs = base_ptrs[:, None] + reduce_offsets[None, :] * stride_reduce_i64
        vals = tl.load(
            x_ptr + ptrs,
            mask=inner_mask[:, None] & mask_reduce[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals, axis=1)

    out_offsets = tl.cast(pid_outer, tl.int64) * inner_size_i64 + inner_offsets
    tl.atomic_add(out_ptr + out_offsets, acc, mask=inner_mask)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim += x.ndim
    x = x.contiguous()
    reduce_size = x.shape[dim]
    out_shape = list(x.shape)
    out_shape[dim] = 1
    if reduce_size == 0:
        return torch.zeros(out_shape, device=x.device, dtype=x.dtype)
