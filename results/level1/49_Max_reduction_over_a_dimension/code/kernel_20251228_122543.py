import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_partial_kernel(
    x_ptr,            # *f32 or *f16
    partial_ptr,      # *same dtype as x
    M, N,             # rows, reduction length
    stride_xm, stride_xn,
    stride_pm, stride_pn,
    BLOCK_N: tl.constexpr,
):
    """
    First-stage reduction: compute per-block maxima along the last dimension.
    Grid:
      axis 0: row index in [0, M)
      axis 1: block index along N in [0, num_blocks)
    """
    pid_m = tl.program_id(axis=0)
    pid_bn = tl.program_id(axis=1)

    if pid_m >= M:
        return

    # Row pointer
    x_row_ptr = x_ptr + pid_m * stride_xm

    # Offsets within the reduction dimension for this block
    offs_n = pid_bn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < N

    vals = tl.load(x_row_ptr + offs_n * stride_xn, mask=mask, other=-float("inf"))
    block_max = tl.max(vals, axis=0)  # scalar

    # Store partial max to [M, num_blocks]
    partial_row_ptr = partial_ptr + pid_m * stride_pm
    tl.store(partial_row_ptr + pid_bn * stride_pn, block_max)


@triton.jit
def max_final_kernel(
    partial_ptr,      # *f32 or *f16
    out_ptr,          # *same dtype as partial
    M,                # number of rows
    stride_pm, stride_pn,
    NUM_BLOCKS: tl.constexpr,  # number of partial blocks per row
    BLOCK_B: tl.constexpr,     # block width for final reduction
):
    """
    Second-stage reduction: reduce partial maxima for each row into a single max.
    Grid:
      axis 0: row index in [0, M)
    """
    pid_m = tl.program_id(axis=0)
    if pid_m >= M:
        return

    row_ptr = partial_ptr + pid_m * stride_pm

    # Accumulator in FP32 for numeric robustness; scalar (not block)
    running_max = tl.full((), -float("inf"), tl.float32)

    # Loop over partial blocks in chunks of BLOCK_B (compile-time constant)
    for start in range(0, NUM_BLOCKS, BLOCK_B):
        offs = start + tl.arange(0, BLOCK_B)
        mask = offs < NUM_BLOCKS

        vals = tl.load(row_ptr + offs * stride_pn, mask=mask, other=-float("inf"))
        vals_f32 = vals.to(tl.float32)
        block_max = tl.max(vals_f32, axis=0)  # scalar
        running_max = tl.maximum(running_max, block_max)

    # Store result; Triton will cast to out_ptr dtype if needed
    tl.store(out_ptr + pid_m, running_max)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    High-performance max reduction along an arbitrary dimension using Triton.
    Equivalent to torch.max(x, dim=dim)[0].
    """
    assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernel"

    ndim = x.dim()
    if ndim == 0:
        # Scalar: max over no dimension is itself (PyTorch doesn't support dim on scalars)
        return x

    dim = dim if dim >= 0 else dim + ndim
    if not (0 <= dim < ndim):
        raise ValueError(f"Invalid dim={dim} for tensor of dim {ndim}")

    # If there are no elements or the outer volume is zero, fall back to PyTorch
    if x.numel() == 0:
        return torch.max(x, dim=dim)[0]

    # Move the reduction dimension to the last axis, keeping others in order
    perm = [i for i in range(ndim) if i != dim] + [dim]
    x_perm = x.permute(perm).contiguous()

    # Make it 2D: [M, N] where N is the reduction length
    N = x_perm.shape[-1]
    if N == 0:
        # PyTorch raises for reduction over empty dimension; defer to it
        return torch.max(x, dim=dim)[0]

    M = x_perm.numel() // N
    if M == 0:
        # Nothing to compute, rely on PyTorch semantics
        return torch.max(x, dim=dim)[0]

    x_2d = x_perm.view(M, N)

    # Strides for row-major [M, N]
    stride_xm = x_2d.stride(0)
    stride_xn = x_2d.stride(1)

    # First-stage: partial maxima along N in blocks of BLOCK_N
    BLOCK_N = 256  # power-of-2 as required
    num_blocks = triton.cdiv(N, BLOCK_N)
    num_blocks = max(num_blocks, 1)

    partial = torch.empty((M, num_blocks), device=x.device, dtype=x.dtype)
    stride_pm = partial.stride(0)
    stride_pn = partial.stride(1)

    grid_partial = lambda meta: (M, num_blocks)
    max_partial_kernel[grid_partial](
        x_2d,
        partial,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_pm,
        stride_pn,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    # Second-stage: reduce partial maxima across num_blocks
    out_flat = torch.empty((M,), device=x.device, dtype=x.dtype)
    NUM_BLOCKS = num_blocks
    BLOCK_B = 256  # power-of-2 as required

    grid_final = lambda meta: (M,)
    max_final_kernel[grid_final](
        partial,
        out_flat,
        M,
        stride_pm,
        stride_pn,
        NUM_BLOCKS=NUM_BLOCKS,
        BLOCK_B=BLOCK_B,
        num_warps=4,
    )

    # Reshape back to original output shape: original tensor with `dim` removed
    out_shape = x_perm.shape[:-1]
    out = out_flat.view(*out_shape)
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
