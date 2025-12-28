import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_dim1_kernel(
    x_ptr,            # *f32
    y_ptr,            # *f32
    B,                # int32: outer dimension
    R,                # int32: reduction dimension
    N,                # int32: inner dimension
    stride_b,         # int32: x.stride(0)
    stride_r,         # int32: x.stride(1)
    stride_n,         # int32: x.stride(2)
    out_stride_b,     # int32: y.stride(0)
    out_stride_r,     # int32: y.stride(1) (reduced dim, size 1)
    out_stride_n,     # int32: y.stride(2)
    BLOCK_R: tl.constexpr,  # power-of-2 block size along reduction dim
):
    pid = tl.program_id(axis=0)
    num_rows = B * N
    if pid >= num_rows:
        return

    # Map linear row id -> (b, n)
    b = pid // N
    n = pid % N

    # Base pointer for this (b, n) slice at r = 0
    base_in = x_ptr + b * stride_b + n * stride_n

    acc = 0.0

    # Loop over reduction dimension in chunks of BLOCK_R
    for r_start in range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        mask = r_offsets < R

        ptrs = base_in + r_offsets * stride_r
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Store result at (b, 0, n) in output
    out_ptr = y_ptr + b * out_stride_b + 0 * out_stride_r + n * out_stride_n
    tl.store(out_ptr, acc)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sum reduction with keepdim=True implemented via Triton.

    Supports arbitrary tensor rank by reshaping to (outer, R, inner)
    and always reducing along the middle dimension.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    if x.numel() == 0:
        # Fallback to PyTorch for degenerate case
        return torch.sum(x, dim=dim, keepdim=True)

    orig_shape = x.shape
    ndim = x.ndim

    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim

    # Make contiguous for predictable strides
    x_contig = x.contiguous()

    # Compute outer, R (reduction size), inner
    sizes = list(orig_shape)
    R = sizes[dim]
    outer = 1
    for i in range(dim):
        outer *= sizes[i]
    inner = 1
    for i in range(dim + 1, ndim):
        inner *= sizes[i]

    # Reshape to (outer, R, inner)
    x_reshaped = x_contig.view(outer, R, inner)

    # Output has shape (outer, 1, inner)
    y_reshaped = torch.empty(
        (outer, 1, inner),
        device=x.device,
        dtype=x.dtype,
    )

    B = outer
    N = inner

    stride_b, stride_r, stride_n = x_reshaped.stride()
    out_stride_b, out_stride_r, out_stride_n = y_reshaped.stride()

    num_rows = B * N
    BLOCK_R = 256  # power-of-2 as required and good for 4096-size reductions

    grid = lambda meta: (triton.cdiv(num_rows, 1),)

    sum_reduce_dim1_kernel[grid](
        x_reshaped,
        y_reshaped,
        B,
        R,
        N,
        stride_b,
        stride_r,
        stride_n,
        out_stride_b,
        out_stride_r,
        out_stride_n,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )

    # Reshape back to original shape with keepdim=True
    out_shape = list(orig_shape[:dim]) + [1] + list(orig_shape[dim + 1 :])
    y = y_reshaped.view(*out_shape)
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_keepdim(x, self.dim)
