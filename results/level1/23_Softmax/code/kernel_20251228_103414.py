import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_2d(
    x_ptr, y_ptr,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Row-wise softmax over dim=1 for a 2D tensor [M, N_COLS].
    Each program handles one row and iterates over it in chunks of BLOCK_SIZE.
    """
    row_id = tl.program_id(axis=0)

    # Base pointers for this row
    row_x_ptr = x_ptr + row_id * stride_xm
    row_y_ptr = y_ptr + row_id * stride_ym

    offs = tl.arange(0, BLOCK_SIZE)

    # --------------------------------------------------------------------- #
    # Pass 1: compute row-wise maximum for numerical stability
    # --------------------------------------------------------------------- #
    row_max = -float("inf")
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        cols = col_start + offs
        mask = cols < N_COLS
        x = tl.load(
            row_x_ptr + cols * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)

    # --------------------------------------------------------------------- #
    # Pass 2: compute exp(x - max) and accumulate row-wise sum
    #         We store the numerators (exp(x - max)) in y_ptr as a temporary.
    # --------------------------------------------------------------------- #
    row_sum = 0.0
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        cols = col_start + offs
        mask = cols < N_COLS
        x = tl.load(
            row_x_ptr + cols * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        x = x - row_max
        exp_x = tl.exp(x)
        row_sum = row_sum + tl.sum(exp_x, axis=0)
        tl.store(
            row_y_ptr + cols * stride_yn,
            exp_x,
            mask=mask,
        )

    # --------------------------------------------------------------------- #
    # Pass 3: normalize stored numerators by the row-wise sum
    # --------------------------------------------------------------------- #
    inv_row_sum = 1.0 / row_sum
    for col_start in range(0, N_COLS, BLOCK_SIZE):
        cols = col_start + offs
        mask = cols < N_COLS
        y = tl.load(
            row_y_ptr + cols * stride_yn,
            mask=mask,
            other=0.0,
        )
        y = y * inv_row_sum
        tl.store(
            row_y_ptr + cols * stride_yn,
            y,
            mask=mask,
        )


def triton_softmax_2d(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton implementation of softmax over dim=1 for 2D tensors.
    """
    assert x.dim() == 2, "triton_softmax_2d expects a 2D tensor"
    assert x.is_cuda, "Input must be a CUDA tensor for Triton execution"

    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_SIZE = 256  # power-of-2 as required

    grid = lambda meta: (M,)  # one program per row; grid size > 0 by construction

    softmax_kernel_2d[grid](
        x, y,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        N_COLS=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Optimized model using a Triton kernel for softmax over dim=1.
    Falls back to PyTorch softmax on non-CUDA tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return triton_softmax_2d(x)
        else:
            return torch.softmax(x, dim=1)
