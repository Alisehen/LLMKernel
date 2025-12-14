import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SEQ': 32}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SEQ': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SEQ': 128}, num_warps=4, num_stages=2),
    ],
    key=['seq_count'],
)
@triton.jit
def cumprod_kernel(
    x_ptr,
    y_ptr,
    axis_len,
    inner,
    seq_count,
    BLOCK_SEQ: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    seq_ids = pid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_seq = seq_ids < seq_count
    if not tl.any(mask_seq):
        return

    axis_len_val = axis_len
    if axis_len_val <= 0:
        return

    inner_val = inner
    outer_idx = seq_ids // inner_val
    inner_idx = seq_ids - outer_idx * inner_val
    base_offset = outer_idx * axis_len_val * inner_val + inner_idx

    offsets = base_offset
    running = tl.load(x_ptr + offsets, mask=mask_seq, other=1)
    tl.store(y_ptr + offsets, running, mask=mask_seq)

    i = 1
    while i < axis_len_val:
        offsets += inner_val
        vals = tl.load(x_ptr + offsets, mask=mask_seq, other=1)
        running = running * vals
        tl.store(y_ptr + offsets, running, mask=mask_seq)
        i += 1


def triton_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("Input must be a CUDA tensor.")
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    dim = dim if dim >= 0 else dim + x_contig.ndim
    axis_len = x_contig.shape[dim]
    if axis_len == 0:
        return out

    inner = x_contig.stride(dim)
    seq_count = x_contig.numel() // axis_len
    grid = ((seq_count + cumprod_kernel.BLOCK_SEQ - 1) // cumprod_kernel.BLOCK_SEQ,)

    cumprod_kernel[grid](
        x_contig,
        out,
        axis_len,
        inner,
        seq_count,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)
