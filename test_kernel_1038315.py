import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    ],
    key=["n_outputs"],
)
@triton.jit
def sum_reduce_dim_kernel(
    x_ptr,          # *f32
    y_ptr,          # *f32
    outer,          # int
    reduce_len,     # int
    inner,          # int
    n_outputs,      # int = outer * inner
    BLOCK_SIZE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_SEGS: tl.constexpr,
):
    """
    Reduce over the middle dimension of a logical [outer, reduce_len, inner] tensor.

    x is assumed to be contiguous, laid out as:
        idx = outer_idx * (reduce_len * inner) + reduce_idx * inner + inner_idx

    Output y has shape [outer, inner] (flattened), with:
        y[outer_idx, inner_idx] = sum_{r} x[outer_idx, r, inner_idx]
    """

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Use 64-bit indices to safely handle very large tensors
    offs64 = offs.to(tl.int64)
    n_outputs_i64 = tl.full((), n_outputs, dtype=tl.int64)
    mask_out = offs64 < n_outputs_i64

    inner_i64 = tl.full((), inner, dtype=tl.int64)
    reduce_i64 = tl.full((), reduce_len, dtype=tl.int64)

    # Map flattened output index -> (outer_idx, inner_idx)
    inner_idx = offs64 % inner_i64
    outer_idx = offs64 // inner_i64

    # Base index for the start of the reduction line:
    # base = outer_idx * (reduce_len * inner) + inner_idx
    base = outer_idx * reduce_i64 * inner_i64 + inner_idx  # [BLOCK_SIZE]

    # Accumulator for each output element
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Offsets along the reduction dimension within a segment
    r_offsets = tl.arange(0, BLOCK_R)
    r_offsets64 = r_offsets.to(tl.int64)  # [BLOCK_R]

    # Process the reduction dimension in NUM_SEGS segments of length BLOCK_R
    for seg in range(NUM_SEGS):
        base_r = seg * BLOCK_R
        idx_r = base_r + r_offsets64  # [BLOCK_R]

        # Broadcast to 2D: [BLOCK_SIZE, BLOCK_R]
        idx_r_b = idx_r[None, :]      # [1, BLOCK_R]
        base_b = base[:, None]        # [BLOCK_SIZE, 1]

        # Pointers for this segment:
        ptrs = x_ptr + base_b + idx_r_b * inner_i64

        # Mask: valid output elements & valid reduction indices
        mask_r = idx_r_b < reduce_i64            # [1, BLOCK_R]
        mask_full = mask_out[:, None] & mask_r   # [BLOCK_SIZE, BLOCK_R]

        # Load and accumulate
        vals = tl.load(ptrs, mask=mask_full, other=0.0)
        seg_sum = tl.sum(vals, axis=1)  # sum over reduction axis in this segment
        acc += seg_sum

    # Compute output offsets (flattened [outer, inner])
    out_offsets = outer_idx * inner_i64 + inner_idx
    tl.store(y_ptr + out_offsets, acc, mask=mask_out)


def triton_sum_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sum reduction over the given dimension with keepdim=True, using Triton.
    Assumes x is contiguous.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernel.")

    x_contig = x.contiguous()
    shape = list(x_contig.shape)
    ndim = len(shape)

    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Invalid reduction dim={dim} for tensor of dim={ndim}")

    # Compute outer, reduce_len, inner for logical [outer, reduce_len, inner] view
    outer = 1
    for i in range(dim):
        outer *= shape[i]

    reduce_len = shape[dim]

    inner = 1
    for i in range(dim + 1, ndim):
        inner *= shape[i]

    n_outputs = outer * inner

    # Allocate output tensor with keepdim=True shape
    out_shape = list(shape)
    out_shape[dim] = 1
    y = torch.empty(out_shape, device=x_contig.device, dtype=x_contig.dtype)

    if n_outputs == 0:
        # Degenerate case, just rely on PyTorch
        return x_contig.sum(dim=dim, keepdim=True)

    # Launch Triton kernel
    BLOCK_R = 128  # reduction tile size
    NUM_SEGS = triton.cdiv(reduce_len, BLOCK_R)

    def grid(meta):
        return (triton.cdiv(n_outputs, meta["BLOCK_SIZE"]),)

    sum_reduce_dim_kernel[grid](
        x_contig,
        y,
        outer,
        reduce_len,
        inner,
        n_outputs,
        BLOCK_R=BLOCK_R,
        NUM_SEGS=NUM_SEGS,
    )

    return y


class ModelNew(nn.Module):
    """
    Model that performs sum reduction over a specified dimension using a Triton kernel.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_reduce(x, self.dim)
