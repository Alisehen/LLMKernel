import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_reduce_dim_kernel(
    x_ptr,       # *f32
    out_ptr,     # *f32
    M,           # number of outer batches
    R,           # reduction length
    N,           # number of inner elements
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)  # along N
    pid_m = tl.program_id(axis=1)  # along M

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    m = pid_m

    # masks for valid indices
    mask_n = n_offsets < N
    mask_m = m < M
    mask = mask_m & mask_n

    # base linear index for (m, r=0, n_offsets)
    base = m * R * N + n_offsets

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    r = 0
    # loop over reduction dimension
    while r < R:
        offs = base + r * N
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += x
        r += 1

    # write result at (m, 0, n_offsets)
    out_base = m * N + n_offsets  # since out shape is [M, 1, N], contiguous
    tl.store(out_ptr + out_base, acc, mask=mask)


def triton_sum_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum reduction over a given dim with keepdim=True using Triton."""
    # Fallback for non-CUDA or unsupported dtypes
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return torch.sum(x, dim=dim, keepdim=True)

    ndim = x.ndim
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim

    orig_shape = list(x.shape)

    # Flatten to [M, R, N] where we reduce over R (axis=1)
    if dim == 0:
        M = 1
        R = orig_shape[0]
        N = int(torch.prod(torch.tensor(orig_shape[1:], device='cpu'))) if ndim > 1 else 1
    elif dim == ndim - 1:
        M = int(torch.prod(torch.tensor(orig_shape[:-1], device='cpu'))) if ndim > 1 else 1
        R = orig_shape[-1]
        N = 1
    else:
        M = int(torch.prod(torch.tensor(orig_shape[:dim], device='cpu')))
        R = orig_shape[dim]
        N = int(torch.prod(torch.tensor(orig_shape[dim + 1:], device='cpu'))) if dim + 1 < ndim else 1

    # Ensure contiguous before reshape for predictable layout
    x_contig = x.contiguous()
    x_reshaped = x_contig.reshape(M, R, N)

    out_reshaped = torch.empty((M, 1, N), device=x.device, dtype=torch.float32)

    BLOCK_N = 128

    def grid(meta):
        return (
            max(1, triton.cdiv(N, meta['BLOCK_N'])),
            max(1, M),
        )

    sum_reduce_dim_kernel[grid](
        x_reshaped,
        out_reshaped,
        M,
        R,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Cast back to original dtype if needed
    if x.dtype != torch.float32:
        out_reshaped = out_reshaped.to(x.dtype)

    # Reshape back to original shape with reduced dim kept as size 1
    out_shape = orig_shape.copy()
    out_shape[dim] = 1
    return out_reshaped.reshape(out_shape)


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sum_keepdim(x, self.dim)
