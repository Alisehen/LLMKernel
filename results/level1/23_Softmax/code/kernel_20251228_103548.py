import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_row_kernel_streaming(
    x_ptr,           # *f32/f16/bf16, input [M, N]
    y_ptr,           # same type as x, output [M, N]
    stride_row,      # row stride for x/y
    stride_col,      # col stride for x/y (usually 1)
    n_cols,          # N
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(axis=0)
    row_start_x = x_ptr + pid * stride_row
    row_start_y = y_ptr + pid * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)

    # -------------------------------------------------------------------------
    # Pass 1: online (streaming) numerically-stable softmax statistics
    #         Compute row_max and row_sum in a single sweep over the row.
    # -------------------------------------------------------------------------
    row_max = -float("inf")
    row_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols

        x = tl.load(row_start_x + cols * stride_col,
                    mask=mask,
                    other=-float("inf"))
        x = x.to(tl.float32)

        # Max over this block
        block_max = tl.max(x, axis=0)

        # New global max for the row
        new_row_max = tl.maximum(row_max, block_max)

        # Rescale old sum to the new max and add contributions from this block
        scale = tl.exp(row_max - new_row_max)
        shifted = x - new_row_max
        exp_shifted = tl.exp(shifted)
        block_exp_sum = tl.sum(exp_shifted, axis=0)

        row_sum = row_sum * scale + block_exp_sum
        row_max = new_row_max

    # -------------------------------------------------------------------------
    # Pass 2: compute final softmax values using the finalized row_max/row_sum
    # -------------------------------------------------------------------------
    inv_row_sum = 1.0 / row_sum

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < n_cols

        x = tl.load(row_start_x + cols * stride_col,
                    mask=mask,
                    other=-float("inf"))
        x_f32 = x.to(tl.float32)

        out = tl.exp(x_f32 - row_max) * inv_row_sum

        # Cast back to original dtype
        out = out.to(x.dtype)
        tl.store(row_start_y + cols * stride_col, out, mask=mask)


def triton_softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over dim=1 for a 2D tensor using a highly optimized Triton kernel.
    Matches torch.softmax(x, dim=1).
    """
    assert x.dim() == 2, "Input must be 2D (batch_size, num_features)"
    x = x.contiguous()
    M, N = x.shape

    y = torch.empty_like(x)

    stride_row, stride_col = x.stride()
    BLOCK_SIZE = 256  # power-of-two, good trade-off for large N

    # One program per row
    grid = lambda meta: (triton.cdiv(M, 1),)

    softmax_row_kernel_streaming[grid](
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
    Behavior matches torch.softmax(x, dim=1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax_dim1(x)
