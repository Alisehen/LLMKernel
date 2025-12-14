import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    n_slices,
    reduce_size,
    inner_size,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_slices:
        return

    pid_i64 = pid.to(tl.int64)
    inner_size_i64 = tl.full((), inner_size, tl.int64)
    reduce_size_i64 = tl.full((), reduce_size, tl.int64)

    inner_idx = (pid % inner_size).to(tl.int64)
    outer_idx = (pid // inner_size).to(tl.int64)

    slice_base = outer_idx * reduce_size_i64 * inner_size_i64 + inner_idx
    k_range = tl.arange(0, BLOCK_K)

    best_val = tl.full((), -float("inf"), tl.float32)
    best_idx = tl.full((), 0, tl.int32)

    k = 0
    while k < reduce_size:
        offsets = k + k_range
        mask = offsets < reduce_size
        offsets_i64 = offsets.to(tl.int64)
        ptrs = slice_base + offsets_i64 * inner_size_i64

        vals = tl.load(x_ptr + ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        current_max = tl.max(vals, axis=0)
        current_arg = tl.argmax(vals, axis=0)
        current_idx = current_arg + k

        better = current_max > best_val
        best_val = tl.where(better, current_max, best_val)
        best_idx = tl.where(better, current_idx, best_idx)

        k += BLOCK_K

    tl.store(out_ptr + pid_i64, best_idx.to(tl.int64))


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x.contiguous()
    ndim = x.ndim
    if dim < 0:
        dim += ndim

    reduce_size = x.shape[dim]

    outer = 1
    for i in range(dim):
        outer *= x.shape[i]

    inner = 1
    for i in range(dim + 1, ndim):
        inner *= x.shape[i]

    out_shape = x.shape[:dim] + x.shape[dim + 1 :]
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    n_slices = out.numel()
    if n_slices == 0:
        return out

    BLOCK_K = 256
    grid = (triton.cdiv(n_slices, 1),)
    argmax_kernel[grid](
        x,
        out.view(-1),
        n_slices,
        reduce_size,
        inner,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, self.dim)
