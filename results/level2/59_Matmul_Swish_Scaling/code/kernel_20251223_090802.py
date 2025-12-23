# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=4,
            num_stages=2,
        ),
        # Wider M, narrower N
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=2,
        ),
        # More warps
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=2,
        ),
        # Aggressive
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N'],
)
@triton.jit
def swish_scale_bias_kernel_2d(
    y_ptr,          # [M, N], row-major
    bias_ptr,       # [N]
    M, N,           # int
    stride_y_row,   # int
    stride_y_col,   # int
    scale,          # scalar (Python float passed in)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile indices
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks
    mask_m = rm < M
    mask_n = cn < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Offsets for y
    offs_y = rm[:, None] * stride_y_row + cn[None, :] * stride_y_col

    # Load y tile
    y = tl.load(y_ptr + offs_y, mask=mask, other=0.0)

    # Load bias (1D)
    bias = tl.load(bias_ptr + cn, mask=mask_n, other=0.0)

    # Ensure all scalar constants follow y's dtype (important for fp16/bf16)
    one = tl.full((), 1.0, dtype=y.dtype)
    # Scale value as same dtype as y
    scale_val = tl.full((), scale, dtype=y.dtype)

    # Bias add
    x = y + bias[None, :]

    # Swish: x * sigmoid(x)
    neg_x = -x
    sig = one / (one + tl.exp(neg_x))
    out = x * sig * scale_val

    # Store back
    tl.store(y_ptr + offs_y, out, mask=mask)


def linear_swish_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (nn.Linear.weight, out_features x in_features)
    bias:   [N]
    scale:  scalar float

    Returns:
        y: [M, N] = (x @ weight.T + bias) * sigmoid(x @ weight.T + bias) * scale
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"

    # GEMM (highly optimized by PyTorch/cuBLAS)
    y = torch.matmul(x, weight.t())

    # Ensure contiguous for Triton
    y = y.contiguous()
    M, N = y.shape
    stride_y_row, stride_y_col = y.stride()

    # Early exit: nothing to do for empty tensors (also avoids 0-size grid)
    if M == 0 or N == 0:
        return y

    # 2D grid over [M, N], ensuring grid dims > 0
    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta['BLOCK_M'])),
            max(1, triton.cdiv(N, meta['BLOCK_N'])),
        )

    swish_scale_bias_kernel_2d[grid](
        y,
        bias,
        M,
        N,
        stride_y_row,
        stride_y_col,
        float(scale),
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the model:

    forward(x):
        y = x @ W^T + b
        y = y * sigmoid(y)    # Swish
        y = y * scaling_factor

    The epilogue (bias + Swish + scale) is fused into a single
    Triton kernel for bandwidth efficiency.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_swish_scale(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scaling_factor,
        )
