import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def argmax_dim1_kernel(
    x_ptr,
    out_ptr,
    B,
    M: tl.constexpr,
    N,
    stride_b,
    stride_m,
    stride_n,
    out_stride_b,
    out_stride_n,
    BLOCK_N: tl.constexpr,
):
    """
    Argmax over dim=1 (M) for a 3D tensor of shape [B, M, N].
    Each program instance processes one batch index `b` and a block of N columns.
    """
    pid_n = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # Guard batch dimension (in case grid is larger than B)
    if pid_b >= B:
        return

    # Columns handled by this program
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # Base offset for this batch
    b_offset = pid_b * stride_b

    # Initialize with row m = 0
    row_offset0 = b_offset  # + 0 * stride_m
    offsets0 = row_offset0 + n_offsets * stride_n
    best_vals = tl.load(x_ptr + offsets0, mask=mask_n, other=-float("inf"))
    best_idx = tl.zeros([BLOCK_N], dtype=tl.int32)

    # Scan rows m = 1..M-1
    for m in range(1, M):
        row_offset = b_offset + m * stride_m
        offsets = row_offset + n_offsets * stride_n
        vals = tl.load(x_ptr + offsets, mask=mask_n, other=-float("inf"))
        better = vals > best_vals
        curr_m = tl.full([BLOCK_N], m, dtype=tl.int32)
        best_vals = tl.where(better, vals, best_vals)
        best_idx = tl.where(better, curr_m, best_idx)

    # Store argmax indices, output shape [B, N]
    out_offsets = pid_b * out_stride_b + n_offsets * out_stride_n
    tl.store(out_ptr + out_offsets, best_idx.to(tl.int64), mask=mask_n)


def triton_argmax_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    Argmax over dim=1 for a 3D CUDA tensor using Triton.
    Expects x with shape [B, M, N]; returns [B, N] with dtype long.
    """
    assert x.dim() == 3, "triton_argmax_dim1 expects a 3D tensor"
    x = x.contiguous()
    B, M, N = x.shape

    out = torch.empty((B, N), device=x.device, dtype=torch.long)

    BLOCK_N = 128  # power-of-2 block size for columns
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)

    argmax_dim1_kernel[grid](
        x,
        out,
        B,
        M,
        N,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_N=BLOCK_N,
    )

    return out


class ModelNew(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension using Triton
    for the common 3D, dim=1 case, with fallback to PyTorch otherwise.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast Triton path: 3D CUDA tensor, argmax over dim=1
        if x.is_cuda and x.dim() == 3 and (self.dim == 1 or self.dim == -2):
            return triton_argmax_dim1(x)
        # Fallback to PyTorch for other cases
        return torch.argmax(x, dim=self.dim)
