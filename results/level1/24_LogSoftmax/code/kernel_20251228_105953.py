import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small / medium feature dims – very common in MLPs
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
        # General strong default for wide dims
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        # Large tiles for very wide rows – best for bandwidth on 4090
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
    ],
    key=['N_COLS'],
)
@triton.jit
def logsoftmax_online_kernel(
    x_ptr,                 # *f32 / *f16
    y_ptr,                 # *f32 / *f16
    N_ROWS,                # number of rows (batch)
    N_COLS: tl.constexpr,  # number of columns (feature dim), compile-time
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax over dim=1 for a 2D tensor of shape (N_ROWS, N_COLS).

    Two code paths:
    - Small-row path (N_COLS <= BLOCK_SIZE): single-pass, keep the row in registers.
    - General online path (N_COLS > BLOCK_SIZE): streaming online log-sum-exp in tiles.
    """
    pid = tl.program_id(axis=0)  # row index
    if pid >= N_ROWS:
        return

    row_start = pid * N_COLS
    offs = tl.arange(0, BLOCK_SIZE)

    # Fast path: entire row fits in a single block -> single global read
    if N_COLS <= BLOCK_SIZE:
        idx = offs
        mask = idx < N_COLS

        # Load entire row once
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))

        # Standard log-sum-exp reduction for this row
        row_max = tl.max(x, axis=0)
        x_centered = x - row_max
        exp_x = tl.exp(x_centered)
        row_sum = tl.sum(exp_x, axis=0)
        logsumexp = row_max + tl.log(row_sum)

        out = x - logsumexp
        tl.store(y_ptr + row_start + idx, out, mask=mask)
    else:
        # General path: online log-sum-exp in tiles.
        # Online state: running max (m) and scaled sum (s).
        m = -float("inf")
        s = 0.0

        # Pass 1: compute per-row logsumexp in a streaming fashion
        for col_start in range(0, N_COLS, BLOCK_SIZE):
            idx = col_start + offs
            mask = idx < N_COLS

            x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))

            # Block-wise log-sum-exp for this tile
            block_max = tl.max(x, axis=0)
            x_centered = x - block_max
            exp_x = tl.exp(x_centered)
            block_sum = tl.sum(exp_x, axis=0)

            # Merge tile (block_max, block_sum) into running (m, s)
            new_m = tl.maximum(m, block_max)
            s = s * tl.exp(m - new_m) + block_sum * tl.exp(block_max - new_m)
            m = new_m

        logsumexp = tl.log(s) + m

        # Pass 2: compute log_softmax = x - logsumexp and write output
        for col_start in range(0, N_COLS, BLOCK_SIZE):
            idx = col_start + offs
            mask = idx < N_COLS

            x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))
            out = x - logsumexp
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

    # One program instance per row (batch dimension).
    grid = lambda META: (triton.cdiv(batch, 1),)

    # N_COLS is compile-time to enable loop unrolling & autotune keying.
    logsoftmax_online_kernel[grid](
        x,
        y,
        batch,
        N_COLS=dim,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs LogSoftmax activation along an arbitrary dim.
    Falls back to reshaping so that the reduction is along the last dimension,
    then applies a fast 2D Triton log_softmax kernel.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match torch.nn.LogSoftmax semantics: apply along self.dim
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)

        dim = self.dim
        if dim < 0:
            dim += x.ndim
        assert 0 <= dim < x.ndim, "Invalid dim for input tensor"

        # Fast path: 2D tensor, reduction along dim=1
        if x.ndim == 2 and dim == 1:
            return triton_logsoftmax_2d(x)

        # General path: move target dim to last, flatten leading dims -> (batch, dim)
        x_moved = x.movedim(dim, -1)
        orig_shape = x_moved.shape
        batch = x_moved.numel() // orig_shape[-1]
        feat_dim = orig_shape[-1]

        x_2d = x_moved.contiguous().view(batch, feat_dim)
        y_2d = triton_logsoftmax_2d(x_2d)
        y_moved = y_2d.view(orig_shape).movedim(-1, dim)

        return y_moved
