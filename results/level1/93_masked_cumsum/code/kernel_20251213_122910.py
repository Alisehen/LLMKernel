import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 4, "num_stages": 3}),
        triton.Config({"num_warps": 8, "num_stages": 2}),
        triton.Config({"num_warps": 8, "num_stages": 3}),
    ],
    key=["rows"],
)
@triton.jit
def masked_chunk_scan_kernel(
    x_ptr,
    mask_ptr,
    partial_ptr,
    chunk_sums_ptr,
    rows,
    cols,
    n_chunks,
    stride_row,
    BLOCK_COL: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if (row >= rows) or (chunk >= n_chunks):
        return

    chunk_start = chunk * BLOCK_COL
    base_offset = row * stride_row
    running = tl.zeros((), dtype=tl.float32)
    chunk_has_work = chunk_start < cols

    for i in tl.static_range(0, BLOCK_COL):
        col_idx = chunk_start + i
        element_active = (col_idx < cols) & chunk_has_work
        x_val = tl.load(x_ptr + base_offset + col_idx, mask=element_active, other=0.0).to(tl.float32)
        m_val = tl.load(mask_ptr + base_offset + col_idx, mask=element_active, other=0.0).to(tl.float32)
        running += x_val * m_val
        tl.store(partial_ptr + base_offset + col_idx, running, mask=element_active)

    tl.store(chunk_sums_ptr + row * n_chunks + chunk, running, mask=chunk_has_work)


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 4, "num_stages": 3}),
        triton.Config({"num_warps": 2, "num_stages": 2}),
    ],
    key=["rows"],
)
@triton.jit
def chunk_prefix_kernel(
    chunk_sums_ptr,
    chunk_offsets_ptr,
    rows,
    n_chunks,
    MAX_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    base = row * n_chunks
    running = tl.zeros((), dtype=tl.float32)

    for idx in tl.static_range(0, MAX_CHUNKS):
        valid = idx < n_chunks
        val = tl.load(chunk_sums_ptr + base + idx, mask=valid, other=0.0).to(tl.float32)
        tl.store(chunk_offsets_ptr + base + idx, running, mask=valid)
        running += val


@triton.autotune(
    configs=[
        triton.Config({"num_warps": 4, "num_stages": 2}),
        triton.Config({"num_warps": 4, "num_stages": 3}),
        triton.Config({"num_warps": 8, "num_stages": 2}),
        triton.Config({"num_warps": 8, "num_stages": 3}),
    ],
    key=["rows"],
)
@triton.jit
def add_chunk_offsets_kernel(
    partial_ptr,
    chunk_offsets_ptr,
    out_ptr,
    rows,
    cols,
    n_chunks,
    stride_row,
    BLOCK_COL: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if (row >= rows) or (chunk >= n_chunks):
        return

    chunk_start = chunk * BLOCK_COL
    offs = chunk_start + tl.arange(0, BLOCK_COL)
    mask = offs < cols
    base_offset = row * stride_row

    vals = tl.load(partial_ptr + base_offset + offs, mask=mask, other=0.0)
    offset = tl.load(chunk_offsets_ptr + row * n_chunks + chunk, mask=chunk_start < cols, other=0.0)
    vals += offset
    tl.store(out_ptr + base_offset + offs, vals, mask=mask)


def _move_dim_last(tensor: torch.Tensor, dim: int):
    dim = dim % tensor.ndim
    permute_order = [i for i in range(tensor.ndim) if i != dim] + [dim]
    inv_permute = [0] * tensor.ndim
    for i, p in enumerate(permute_order):
        inv_permute[p] = i
    tensor_perm = tensor.permute(permute_order).contiguous()
    cols = tensor_perm.shape[-1]
    rows = tensor_perm.numel() // cols
    tensor_2d = tensor_perm.view(rows, cols)
    return tensor_2d, tensor_perm.shape, inv_permute


def masked_cumsum_triton(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    x = x.contiguous()
    mask = mask.contiguous()

    x_2d, perm_shape, inv_perm = _move_dim_last(x, dim)
    mask_2d, _, _ = _move_dim_last(mask, dim)

    rows, cols = x_2d.shape
    out_2d = torch.empty_like(x_2d)
    partial = torch.empty_like(x_2d)

    BLOCK_COL = 128
    MAX_CHUNKS = 1024
    n_chunks = triton.cdiv(cols, BLOCK_COL)
    if n_chunks > MAX_CHUNKS:
        raise ValueError(f"Number of chunks {n_chunks} exceeds MAX_CHUNKS={MAX_CHUNKS}")

    chunk_sums = torch.zeros((rows, n_chunks), dtype=x.dtype, device=x.device)
    chunk_offsets = torch.zeros_like(chunk_sums)
    stride_row = x_2d.stride(0)
    grid_blocks = (rows, n_chunks)

    masked_chunk_scan_kernel[grid_blocks](
        x_2d, mask_2d, partial, chunk_sums,
        rows, cols, n_chunks, stride_row,
        BLOCK_COL=BLOCK_COL,
    )

    chunk_prefix_kernel[(rows,)](
        chunk_sums, chunk_offsets,
        rows, n_chunks,
        MAX_CHUNKS=MAX_CHUNKS,
    )

    add_chunk_offsets_kernel[grid_blocks](
        partial, chunk_offsets, out_2d,
        rows, cols, n_chunks, stride_row,
        BLOCK_COL=BLOCK_COL,
    )

    out_perm = out_2d.view(perm_shape).permute(inv_perm).contiguous()
    return out_perm


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum_triton(x, mask, self.dim)
