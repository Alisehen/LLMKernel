import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_row_kernel(
    x_ptr,           # pointer to input [M, N]
    y_ptr,           # pointer to output [M, N]
    stride_row,      # row stride for x/y
    stride_col,      # col stride for x/y (usually 1)
    n_cols,          # N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # each program handles one row
    row_start_x = x_ptr + pid * stride_row
    row_start_y = y_ptr + pid * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)

    # 1) compute row-wise max for numerical stability
    row_max = -float("inf")
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        x = tl.load(row_start_x + cols * stride_col, mask=mask, other=-float("inf"))
        current_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, current_max)

    # 2) compute sum of exp(x - row_max)
    sum_exp = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        x = tl.load(row_start_x + cols * stride_col, mask=mask, other=-float("inf"))
        x = x - row_max
        exp_x = tl.exp(x)
        sum_exp += tl.sum(exp_x, axis=0)

    # 3) normalize to get softmax
    inv_sum = 1.0 / sum_exp
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols
        x = tl.load(row_start_x + cols * stride_col, mask=mask, other=-float("inf"))
        x = x - row_max
        exp_x = tl.exp(x)
        out = exp_x * inv_sum
        tl.store(row_start_y + cols * stride_col, out, mask=mask)


def triton_softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over dim=1 for a 2D tensor using a highly optimized Triton kernel.
    """
    assert x.dim() == 2, "Input must be 2D (batch_size, num_features)"
    # Ensure contiguous layout for predictable strides and coalesced memory access
    x = x.contiguous()
    M, N = x.shape

    y = torch.empty_like(x)

    stride_row, stride_col = x.stride()

    BLOCK_SIZE = 256  # power-of-two, fits constraint and gives good occupancy

    # One program per row
    grid = lambda meta: (triton.cdiv(M, 1),)

    softmax_row_kernel[grid](
        x,
        y,
        stride_row,
        stride_col,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model performing softmax over dim=1.
    Matches the behavior of torch.softmax(x, dim=1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep semantics identical to the original model: softmax over dim=1
        return triton_softmax_dim1(x)
