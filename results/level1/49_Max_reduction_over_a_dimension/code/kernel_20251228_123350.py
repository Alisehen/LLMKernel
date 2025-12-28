import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_strided_kernel(
    x_ptr,                  # *dtype
    out_ptr,                # *dtype
    shape_outer_ptr,        # *int32, length = MAX_OUT_DIMS
    stride_outer_ptr,       # *int32, length = MAX_OUT_DIMS
    stride_r,               # int32, stride of reduction dim (last dim of x_perm)
    R,                      # int32, length of reduction dim
    out_numel,              # int32, number of output elements (product of outer dims)
    BLOCK_M: tl.constexpr,  # number of output elements per program
    BLOCK_R: tl.constexpr,  # reduction tile size
    MAX_OUT_DIMS: tl.constexpr,  # length of shape_outer/stride_outer arrays
    NUM_R_BLOCKS: tl.constexpr,  # = ceil_div(R, BLOCK_R)
):
    """
    Single-pass max reduction over the last dimension of an arbitrarily-strided tensor.

    The input tensor is logically shaped as [D0, D1, ..., D_{K-1}, R],
    where R is the reduction dimension (last axis). K = original_ndim - 1.

    Each program instance reduces BLOCK_M output elements (rows), where each
    output element corresponds to a unique combination of indices over
    [D0, ..., D_{K-1}]. For each such combination, we traverse the R dimension
    (possibly strided) in tiles of size BLOCK_R and accumulate the maximum
    entirely in registers.
    """
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < out_numel  # [BLOCK_M]

    # Decode linear output index -> base offset in x (excluding reduction dim)
    # We treat outer dims as a flattened index with mixed radix given by shape_outer.
    linear = offs_m
    base_offsets = tl.zeros([BLOCK_M], dtype=tl.int32)

    # Decompose 'linear' into multi-index over outer dims in reverse order.
    # shape_outer/stride_outer have length MAX_OUT_DIMS; unused dims have size=1, stride=0.
    for rev in tl.static_range(0, MAX_OUT_DIMS):
        dim = MAX_OUT_DIMS - 1 - rev
        size_i = tl.load(shape_outer_ptr + dim)   # scalar int32
        stride_i = tl.load(stride_outer_ptr + dim)
        idx_i = linear % size_i
        linear = linear // size_i
        base_offsets += idx_i * stride_i

    # Running max per output element, in FP32 for numeric robustness
    running_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    # Iterate over reduction dimension in tiles of size BLOCK_R
    for rb in tl.static_range(0, NUM_R_BLOCKS):
        r_idx = rb * BLOCK_R + tl.arange(0, BLOCK_R)  # [BLOCK_R]
        mask_r = r_idx < R                             # [BLOCK_R]

        # Compute pointers for a [BLOCK_M, BLOCK_R] tile
        base = base_offsets[:, None]                   # [BLOCK_M, 1]
        r_offsets = r_idx[None, :] * stride_r          # [1, BLOCK_R]
        ptrs = x_ptr + base + r_offsets                # [BLOCK_M, BLOCK_R]

        mask = mask_m[:, None] & mask_r[None, :]       # [BLOCK_M, BLOCK_R]
        vals = tl.load(ptrs, mask=mask, other=-float("inf"))
        vals_f32 = vals.to(tl.float32)

        # Reduce across the reduction-tile dimension (axis=1)
        block_max = tl.max(vals_f32, axis=1)           # [BLOCK_M]
        running_max = tl.maximum(running_max, block_max)

    # Write results back; Triton will cast to out_ptr dtype if needed
    tl.store(out_ptr + offs_m, running_max, mask=mask_m)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    High-performance max reduction along an arbitrary dimension using a single-pass
    Triton kernel over the original tensor layout (no contiguous transpose, no
    intermediate partial reductions).

    Equivalent to: torch.max(x, dim=dim)[0]
    """
    assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernel"

    ndim = x.dim()
    if ndim == 0:
        # Scalar: same behavior as PyTorch (no dim argument)
        return x

    # Normalize dimension
    dim = dim if dim >= 0 else dim + ndim
    if not (0 <= dim < ndim):
        raise ValueError(f"Invalid dim={dim} for tensor of dim {ndim}")

    # Handle empty tensors or empty reduction dims via PyTorch for correct errors
    if x.numel() == 0 or x.shape[dim] == 0:
        return torch.max(x, dim=dim)[0]

    # Restrict to floating-point dtypes (common for reductions). Fallback otherwise.
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.max(x, dim=dim)[0]

    # Move the reduction dimension to the last axis using a view (no data copy).
    perm = [i for i in range(ndim) if i != dim] + [dim]
    x_perm = x.permute(perm)

    # Logical shape [*outer_shape, R] where we reduce over R (last dim)
    shape_perm = list(x_perm.shape)
    R = shape_perm[-1]
    outer_shape = shape_perm[:-1]

    # Output tensor has the same outer shape (original tensor with `dim` removed)
    out = torch.empty(outer_shape, device=x.device, dtype=x.dtype)
    out_numel = out.numel()

    # Support up to MAX_OUT_DIMS outer dimensions
    MAX_OUT_DIMS = 8
    outer_ndim = len(outer_shape)
    if outer_ndim > MAX_OUT_DIMS:
        # Fallback for very high-rank tensors
        return torch.max(x, dim=dim)[0]

    # Build shape/stride arrays for outer dimensions (padded to MAX_OUT_DIMS)
    shape_outer = torch.ones(MAX_OUT_DIMS, dtype=torch.int32, device=x.device)
    stride_outer = torch.zeros(MAX_OUT_DIMS, dtype=torch.int32, device=x.device)
    for i, size in enumerate(outer_shape):
        shape_outer[i] = size
        stride_outer[i] = x_perm.stride(i)

    stride_r = int(x_perm.stride(-1))

    # Tiling parameters
    BLOCK_M = 128
    BLOCK_R = 128
    NUM_R_BLOCKS = triton.cdiv(R, BLOCK_R)

    # Launch configuration: one axis over output elements
    def grid(meta):
        return (triton.cdiv(out_numel, meta["BLOCK_M"]),)

    max_reduce_strided_kernel[grid](
        x_perm,
        out,
        shape_outer,
        stride_outer,
        stride_r,
        R,
        out_numel,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        MAX_OUT_DIMS=MAX_OUT_DIMS,
        NUM_R_BLOCKS=NUM_R_BLOCKS,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_reduce(x, self.dim)
