import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(
    x_ptr,             # *f32 / *f16
    row_max_ptr,       # *f32 / *f16
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index
    row_start = pid * N_COLS

    offs = tl.arange(0, BLOCK_SIZE)
    max_val = -float("inf")

    # Pass 1: row-wise maximum
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N_COLS
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    tl.store(row_max_ptr + pid, max_val)


@triton.jit
def logsoftmax_kernel(
    x_ptr,             # *f32 / *f16
    row_max_ptr,       # *f32 / *f16
    y_ptr,             # *f32 / *f16
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index
    row_start = pid * N_COLS
    offs = tl.arange(0, BLOCK_SIZE)

    # Load precomputed row max
    max_val = tl.load(row_max_ptr + pid)

    # Pass 1: compute sum(exp(x - max))
    sum_exp = 0.0
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N_COLS
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))
        x_centered = x - max_val
        exp_x = tl.exp(x_centered)
        sum_exp = sum_exp + tl.sum(exp_x, axis=0)

    logsumexp = tl.log(sum_exp)

    # Pass 2: write log_softmax = (x - max) - logsumexp
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N_COLS
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))
        x_centered = x - max_val
        out = x_centered - logsumexp
        tl.store(y_ptr + row_start + idx, out, mask=mask)


def triton_logsoftmax_2d(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton implementation of log_softmax over dimension 1
    for a 2D tensor of shape (batch, dim).
    """
    assert x.ndim == 2, "triton_logsoftmax_2d only supports 2D tensors (batch, dim)"
    assert x.is_cuda, "Input must be on CUDA device for Triton kernels"

    x = x.contiguous()
    batch, dim = x.shape
    y = torch.empty_like(x)

    # Buffer to hold per-row maxima
    row_max = torch.empty(
        batch, device=x.device, dtype=x.dtype
    )

    BLOCK_SIZE = 256

    # Grid along rows; ensure > 0 using cdiv as requested
    grid = lambda meta: (triton.cdiv(batch, 1),)

    # Kernel 1: row-wise max
    row_max_kernel[grid](
        x,
        row_max,
        N_COLS=dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    # Kernel 2: log_softmax using row_max
    logsoftmax_kernel[grid](
        x,
        row_max,
        y,
        N_COLS=dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs LogSoftmax activation along dim=1.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This implementation is specialized for log_softmax over dim=1
        assert self.dim in (1, -1), "ModelNew only supports dim=1 or dim=-1 for now"
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        # Reshape to 2D if necessary (treat all leading dims as batch)
        if x.ndim != 2:
            new_dim = x.shape[self.dim]
            batch = x.numel() // new_dim
            x = x.contiguous().view(batch, new_dim)
            y = triton_logsoftmax_2d(x)
            return y.view(*([d for i, d in enumerate(x.shape) if i != 1] + [x.shape[1]]))
        else:
            return triton_logsoftmax_2d(x)
