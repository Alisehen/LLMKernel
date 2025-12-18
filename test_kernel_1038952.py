import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumprod_chunk_kernel(
    x_ptr,                # *T
    y_ptr,                # *T
    prev_prod_ptr,        # *fp32, per-segment running product
    length,               # int: size along cumprod dimension
    inner,                # int: product of sizes after dim
    n_segments,           # int: number of (outer, inner) segments
    chunk_start,          # int: starting column (along dim) for this chunk
    BLOCK_SEGMENTS: tl.constexpr,  # number of segments processed per program
    CHUNK_COLS: tl.constexpr,      # number of columns (along dim) per call
):
    pid = tl.program_id(axis=0)
    seg_start = pid * BLOCK_SEGMENTS
    seg_offsets = seg_start + tl.arange(0, BLOCK_SEGMENTS)
    mask_seg = seg_offsets < n_segments

    inner_t = inner
    length_t = length

    # Map flat segment index -> base pointer offset
    # Treat original tensor as shape (outer, length, inner) with strides:
    #   (length * inner, inner, 1)
    outer_idx = seg_offsets // inner_t
    inner_idx = seg_offsets % inner_t
    base_offset = outer_idx * length_t * inner_t + inner_idx

    # Load previous cumulative product for each segment
    prev = tl.load(prev_prod_ptr + seg_offsets, mask=mask_seg, other=1.0)
    curr = prev

    # Process a fixed-size chunk of columns along the cumprod dimension
    for j in range(CHUNK_COLS):
        col = chunk_start + j
        col_mask = mask_seg & (col < length_t)
        offset = base_offset + col * inner_t

        x_val = tl.load(x_ptr + offset, mask=col_mask, other=1.0)
        x_val = x_val.to(tl.float32)

        curr = curr * x_val
        tl.store(y_ptr + offset, curr, mask=col_mask)

    # Store updated cumulative product (used as starting point for next chunk)
    tl.store(prev_prod_ptr + seg_offsets, curr, mask=mask_seg)


def triton_cumprod(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Fallback for non-CUDA or trivial cases
    if (not x.is_cuda) or x.numel() == 0:
        return torch.cumprod(x, dim=dim)

    x_contig = x.contiguous()
    shape = x_contig.shape
    ndim = x_contig.dim()

    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise ValueError(f"Invalid dim={dim} for tensor of dim={ndim}")

    # Size along cumprod dimension
    length = shape[dim]
    if length == 0:
        return x_contig.clone()

    # Compute inner = product of sizes after dim, outer = product before dim
    if dim + 1 < ndim:
        inner = int(torch.prod(torch.tensor(shape[dim + 1:], dtype=torch.int64)).item())
    else:
        inner = 1

    if dim > 0:
        outer = int(torch.prod(torch.tensor(shape[:dim], dtype=torch.int64)).item())
    else:
        outer = 1

    n_segments = outer * inner
    if n_segments == 0:
        return x_contig.clone()

    x_flat = x_contig.view(-1)
    y_flat = torch.empty_like(x_flat)

    # Running products per segment, kept in fp32 for stability
    prev_prod = torch.ones(n_segments, device=x.device, dtype=torch.float32)

    BLOCK_SEGMENTS = 128  # power-of-2, number of segments per program
    CHUNK_COLS = 128      # power-of-2, columns processed per kernel call

    grid = lambda meta: (triton.cdiv(n_segments, meta['BLOCK_SEGMENTS']),)

    # Sweep along the cumprod dimension in fixed-size chunks
    for chunk_start in range(0, length, CHUNK_COLS):
        cumprod_chunk_kernel[grid](
            x_flat, y_flat, prev_prod,
            length, inner, n_segments, chunk_start,
            BLOCK_SEGMENTS=BLOCK_SEGMENTS,
            CHUNK_COLS=CHUNK_COLS,
        )

    y = y_flat.view(shape)
    return y


class ModelNew(nn.Module):
    """
    Triton-backed cumulative product model along a specified dimension.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumprod(x, self.dim)
