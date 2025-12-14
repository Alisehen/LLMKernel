import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    VEC: tl.constexpr,
):
    row = tl.program_id(axis=0)
    chunk = tl.program_id(axis=1)
    if (row >= rows) or (chunk >= n_chunks):
        return

    chunk_start = chunk * BLOCK_COL
    if chunk_start >= cols:
        return

    base_offset = row * stride_row
    row_x_ptr = x_ptr + base_offset
    row_mask_ptr = mask_ptr + base_offset
    row_partial_ptr = partial_ptr + base_offset

    running = tl.zeros((), dtype=tl.float32)
    chunk_offsets = tl.arange(0, VEC)

    for blk in tl.static_range(0, BLOCK_COL, VEC):
        cols_block = chunk_start + blk + chunk_offsets
        block_mask = cols_block < cols
        any_valid = tl.any(block_mask, axis=0)
        if not any_valid:
            break

        safe_cols = tl.where(block_mask, cols_block, chunk_start)
        x_vals = tl.load(row_x_ptr + safe_cols, mask=block_mask, other=0.0).to(tl.float32)
        m_vals = tl.load(row_mask_ptr + safe_cols, mask=block_mask, other=0.0).to(tl.float32)
        prods = x_vals * m_vals

        for lane in range(VEC):
            col_idx = chunk_start + blk + lane
            lane_mask = col_idx < cols
            contrib = tl.where(lane_mask, prods[lane], 0.0)
            running += contrib
            tl.store(row_partial_ptr + col_idx, running, mask=lane_mask)

    chunk_ptr = chunk_sums_ptr + row * n_chunks + chunk
    tl.store(chunk_ptr, running)


@triton.jit
def chunk_prefix_apply_kernel(
    partial_ptr,
    chunk_sums_ptr,
    out_ptr,
    rows,
    cols,
    n_chunks,
    stride_row,
    BLOCK_COL: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= rows:
        return

    row_partial_ptr = partial_ptr + row * stride_row
    row_out_ptr = out_ptr + row * stride_row
    chunk_row_ptr = chunk_sums_ptr + row * n_chunks

    running = tl.zeros((), dtype=tl.float32)
    col_offsets = tl.arange(0, BLOCK_COL)

    for chunk_idx in tl.static_range(0, MAX_CHUNKS):
        chunk_active = chunk_idx < n_chunks
        chunk_start = chunk_idx * BLOCK_COL
        cols_idx = chunk_start + col_offsets
        elem_mask = chunk_active & (cols_idx < cols)

        vals = tl.load(row_partial_ptr + cols_idx, mask=elem_mask, other=0.0)
        vals = vals + running
        vals = vals.to(OUT_DTYPE)
        tl.store(row_out_ptr + cols_idx, vals, mask=elem_mask)

        chunk_sum = tl.load(chunk_row_ptr + chunk_idx, mask=chunk_active, other=0.0)
        running += chunk_sum


def _move_dim_last(tensor: torch.Tensor, dim: int):
    dim = dim % tensor.ndim
    permute_order = [i for i in range(tensor.ndim) if i != dim] + [dim]
    inv_permute = [0] * tensor.ndim
    for i, p in enumerate(permute_order):
        inv_permute[p] = i
    tensor_perm = tensor.permute(permute_order).contiguous()
    shape_perm = tensor_perm.shape
    cols = shape_perm[-1]
    rows = tensor_perm.numel() // cols
    tensor_2d = tensor_perm.view(rows, cols)
    return tensor_2d, shape_perm, inv_permute


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def masked_cumsum_triton(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    x = x.contiguous()
    mask = mask.contiguous()

    x_2d, perm_shape, inv_perm = _move_dim_last(x, dim)
    mask_2d, _, _ = _move_dim_last(mask, dim)

    rows, cols = x_2d.shape
    out_2d = torch.empty_like(x_2d)
    partial = torch.empty((rows, cols), dtype=torch.float32, device=x.device)

    BLOCK_COL = 128
    MAX_CHUNKS = 1024
    VEC = 4
    n_chunks = triton.cdiv(cols, BLOCK_COL)
    if n_chunks > MAX_CHUNKS:
        raise ValueError(f"Number of chunks {n_chunks} exceeds MAX_CHUNKS={MAX_CHUNKS}")

    chunk_sums = torch.empty((rows, n_chunks), dtype=torch.float32, device=x.device)

    stride_row = x_2d.stride(0)

    grid_scan = (rows, n_chunks)
    masked_chunk_scan_kernel[grid_scan](
        x_2d,
        mask_2d,
        partial,
        chunk_sums,
        rows,
        cols,
        n_chunks,
        stride_row,
        BLOCK_COL=BLOCK_COL,
        VEC=VEC,
        num_warps=4,
        num_stages=2,
    )

    grid_rows = (rows,)
    chunk_prefix_apply_kernel[grid_rows](
        partial,
        chunk_sums,
        out_2d,
        rows,
        cols,
        n_chunks,
        stride_row,
        BLOCK_COL=BLOCK_COL,
        MAX_CHUNKS=MAX_CHUNKS,
        OUT_DTYPE=_torch_dtype_to_triton(x.dtype),
        num_warps=4,
        num_stages=1,
    )

    out_perm = out_2d.view(perm_shape)
    out = out_perm.permute(inv_perm).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, mask):
        return masked_cumsum_triton(x, mask, self.dim)
