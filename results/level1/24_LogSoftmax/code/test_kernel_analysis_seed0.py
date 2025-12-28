import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    # Guard against extra blocks when grid > M
    if pid_m >= M:
        return

    # Row pointers
    row_x_ptr = x_ptr + pid_m * stride_xm
    row_y_ptr = y_ptr + pid_m * stride_ym

    offsets = tl.arange(0, BLOCK_SIZE)
    n_cols = N

    # -----------------------
    # Pass 1: compute row max
    # -----------------------
    col = 0
    row_max = -float("inf")
    while col < n_cols:
        idx = col + offsets
        mask = idx < n_cols
        x = tl.load(
            row_x_ptr + idx * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)
        col += BLOCK_SIZE

    # ------------------------------------------
    # Pass 2: compute sum(exp(x - row_max)) per row
    # ------------------------------------------
    col = 0
    row_sum = 0.0
    while col < n_cols:
        idx = col + offsets
        mask = idx < n_cols
        x = tl.load(
            row_x_ptr + idx * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        x = x - row_max
        exp_x = tl.exp(x)
        row_sum = row_sum + tl.sum(exp_x, axis=0)
        col += BLOCK_SIZE

    log_row_sum = tl.log(row_sum)

    # ---------------------------------
    # Pass 3: write log_softmax outputs
    # ---------------------------------
    col = 0
    while col < n_cols:
        idx = col + offsets
        mask = idx < n_cols
        x = tl.load(
            row_x_ptr + idx * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        y = x - row_max - log_row_sum
        tl.store(
            row_y_ptr + idx * stride_yn,
            y,
            mask=mask,
        )
        col += BLOCK_SIZE


def triton_log_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    High-performance row-wise log_softmax using Triton.
    Optimized for large last-dimension (e.g., [batch, dim] with dim >> 1).
    Currently supports dim being the last dimension.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device for Triton kernel.")

    # Normalize dim to positive and ensure it's the last dimension
    if dim < 0:
        dim = x.ndim + dim
    if dim != x.ndim - 1:
        raise NotImplementedError("triton_log_softmax only supports softmax over the last dimension.")

    # Collapse all leading dimensions into a single batch dimension
    orig_shape = x.shape
    batch = int(torch.prod(torch.tensor(orig_shape[:-1])).item())
    n_cols = orig_shape[-1]

    x_2d = x.contiguous().view(batch, n_cols)
    y_2d = torch.empty_like(x_2d)

    M, N = x_2d.shape
    BLOCK_SIZE = 256  # power-of-2 as required

    x_ptr = x_2d
    y_ptr = y_2d

    stride_xm, stride_xn = x_2d.stride()
    stride_ym, stride_yn = y_2d.stride()

    grid = lambda meta: (max(M, 1),)

    log_softmax_kernel[grid](
        x_ptr,
        y_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        M,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    return y_2d.view(orig_shape)


class ModelNew(nn.Module):
    """
    Triton-optimized model that performs a LogSoftmax activation along a given dimension.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x, dim=self.dim)
