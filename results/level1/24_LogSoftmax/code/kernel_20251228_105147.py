import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def logsoftmax_online_kernel(
    x_ptr,             # *f32 / *f16
    y_ptr,             # *f32 / *f16
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index
    row_start = pid * N_COLS

    offs = tl.arange(0, BLOCK_SIZE)

    # Online log-sum-exp state: running max (m) and scaled sum (s)
    m = -float("inf")
    s = 0.0

    # Pass 1: single streaming pass to compute logsumexp per row
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        idx = col_start + offs
        mask = idx < N_COLS

        x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float("inf"))

        # Block-wise log-sum-exp
        block_max = tl.max(x, axis=0)
        x_centered = x - block_max
        exp_x = tl.exp(x_centered)
        block_sum = tl.sum(exp_x, axis=0)

        # Merge block (block_max, block_sum) into running (m, s)
        new_m = tl.maximum(m, block_max)
        # rescale old and new sums into the new_max domain
        s_scaled_old = s * tl.exp(m - new_m)
        s_scaled_block = block_sum * tl.exp(block_max - new_m)
        s = s_scaled_old + s_scaled_block
        m = new_m

    # Final logsumexp for the row
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

    BLOCK_SIZE = 256

    # One program instance per row
    grid = lambda meta: (triton.cdiv(batch, 1),)

    logsoftmax_online_kernel[grid](
        x,
        y,
        N_COLS=dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
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
