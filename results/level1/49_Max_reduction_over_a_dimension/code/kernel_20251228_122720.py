import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_reduce_lastdim_kernel(
    x_ptr,            # *f32 / *f16
    out_ptr,          # *f32 / *f16
    M,                # number of rows
    stride_m,         # stride between rows
    stride_k,         # stride along reduction dimension
    stride_out,       # stride for output rows
    K: tl.constexpr,  # reduction size (last dimension)
    BLOCK_SIZE: tl.constexpr,  # tile size along K
):
    pid = tl.program_id(axis=0)
    m = pid  # row index

    # Base pointer for this row
    row_start = x_ptr + m * stride_m

    # Running maximum in FP32 for better accuracy
    running_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)

    # Iterate over the reduction dimension in BLOCK_SIZE chunks
    for k in range(0, K, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        ptrs = row_start + offs * stride_k

        vals = tl.load(ptrs, mask=mask, other=-float("inf"))
        vals_f32 = vals.to(tl.float32)
        running_max = tl.maximum(running_max, vals_f32)

    # Reduce across the BLOCK_SIZE lanes to get a single scalar max
    max_val = tl.max(running_max, axis=0)

    # Store result for this row
    out_ptr_m = out_ptr + m * stride_out
    tl.store(out_ptr_m, max_val)


def triton_max_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Max reduction over a given dimension using a high-performance Triton kernel.
    Returns only the values (same as torch.max(x, dim=dim)[0]).
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16, bf16, fp32"

    ndim = x.ndim
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim

    # Move the reduction dimension to the last position: [other_dims..., dim]
    if dim == ndim - 1:
        x_perm = x.contiguous()
    else:
        perm = [i for i in range(ndim) if i != dim] + [dim]
        x_perm = x.permute(perm).contiguous()

    # Collapse all non-reduced dims into a single leading dimension
    K = x_perm.size(-1)                # reduction size
    M = x_perm.numel() // K            # number of rows
    x_2d = x_perm.view(M, K)

    # Prepare output: one value per row
    out = torch.empty(M, device=x.device, dtype=x.dtype)

    # Strides in elements
    stride_m = x_2d.stride(0)
    stride_k = x_2d.stride(1)
    stride_out = out.stride(0)

    BLOCK_SIZE = 128  # power-of-2 as required

    # One program per row
    grid = lambda meta: (M,)

    max_reduce_lastdim_kernel[grid](
        x_2d,
        out,
        M,
        stride_m,
        stride_k,
        stride_out,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Reshape back to original shape with the reduced dim removed
    out_shape = x_perm.shape[:-1]
    out_view = out.view(*out_shape)

    # The permutation [other_dims..., dim] followed by reduction on last
    # already yields the correct order of remaining dimensions, so no
    # inverse permutation is needed.
    return out_view


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension
    using a high-performance Triton kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_dim(x, self.dim)
