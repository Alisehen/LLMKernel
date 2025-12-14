# import order (torch, torch.nn, triton, triton.language as tl)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
    ],
    key=["axis_len"],
)
@triton.jit
def cumprod_kernel(
    x_ptr,
    y_ptr,
    axis_len,
    inner,
    seq_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= seq_count:
        return

    inner_val = inner
    axis_len_val = axis_len
    if axis_len_val <= 0:
        return

    seq_stride = axis_len_val * inner_val
    outer_idx = pid // inner_val
    inner_idx = pid % inner_val
    base_offset = outer_idx * seq_stride + inner_idx

    first = tl.load(x_ptr + base_offset)
    tl.store(y_ptr + base_offset, first)
    running = first

    start = 1
    while start < axis_len_val:
        idx = start + tl.arange(0, BLOCK)
        mask = idx < axis_len_val
        offs = base_offset + idx * inner_val
        vals = tl.load(x_ptr + offs, mask=mask, other=1)
        out_vals = tl.zeros((BLOCK,), dtype=vals.dtype)
        acc = running
        for i in range(BLOCK):
            m = mask[i]
            val = vals[i]
            acc = tl.where(m, acc * val, acc)
            out_vals[i] = acc
        tl.store(y_ptr + offs, out_vals, mask=mask)
        running = acc
        start += BLOCK


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

    grid = (seq_count,)
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
