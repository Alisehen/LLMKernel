# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def _gelu_logistic_approx(x: tl.tensor) -> tl.tensor:
    # Approximate GELU using logistic CDF:
    #   Φ(x) ≈ 1 / (1 + exp(-1.702 * x))
    #   GELU(x) ≈ x * Φ(x)
    k = 1.702
    p = 1.0 / (1.0 + tl.exp(-k * x))
    return x * p


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_lse_leaky2_gelu2_kernel(
    a_ptr,      # [M, K]
    b_ptr,      # [K, N] = weight.T
    bias_ptr,   # [N] or unused if HAS_BIAS = False
    out_ptr,    # [M, 1]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < M

    # Initialize running values for streaming LogSumExp per row
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Outer loop over N tiles (columns)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N

        # Accumulator for this [BLOCK_M, BLOCK_N] output tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Inner loop over K tiles (reduction dimension)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

            a = tl.load(
                a_ptrs,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )

            acc += tl.dot(a, b, allow_tf32=True)

        # Add bias if present
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=col_mask, other=0.0)
            acc += bias[None, :]

        # Mask out invalid columns so they don't affect max/sum
        acc = tl.where(col_mask[None, :], acc, -float("inf"))

        # Streaming row-wise LogSumExp update for this tile
        tile_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_row_max = tl.maximum(row_max, tile_max)

        exp_tile = tl.exp(acc - new_row_max[:, None])          # [BLOCK_M, BLOCK_N]
        sum_exp_tile = tl.sum(exp_tile, axis=1)                 # [BLOCK_M]

        row_sum = row_sum * tl.exp(row_max - new_row_max) + sum_exp_tile
        row_max = new_row_max

    # Final LogSumExp per row: lse = max + log(sum_exp)
    lse = row_max + tl.log(row_sum)

    # Two LeakyReLU activations
    neg = negative_slope
    y = tl.where(lse >= 0, lse, neg * lse)
    y = tl.where(y >= 0, y, neg * y)

    # Two GELU (approximate) activations
    y = _gelu_logistic_approx(y)
    y = _gelu_logistic_approx(y)

    # Store final scalar result per row -> [M, 1]
    out_ptrs = out_ptr + offs_m * stride_om
    tl.store(out_ptrs, y, mask=row_mask)


def fused_linear_lse_leaky2_gelu2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    negative_slope: float = 0.01,
) -> torch.Tensor:
    """
    Fused kernel:
      y = x @ weight.T + bias
      y = logsumexp(y, dim=1, keepdim=True)
      y = LeakyReLU(y) -> LeakyReLU(y)
      y = GELU_approx(y) -> GELU_approx(y)

    x:      [M, K], float32 CUDA
    weight: [N, K], float32 CUDA (nn.Linear.weight layout)
    bias:   [N],    float32 CUDA or None
    returns: [M, 1], float32 CUDA
    """
    assert x.is_cuda and weight.is_cuda
    assert x.dtype == torch.float32
    assert weight.dtype == torch.float32
    if bias is not None:
        assert bias.is_cuda and bias.dtype == torch.float32

    M, K = x.shape
    N = weight.shape[0]
    # B matrix: [K, N]
    b = weight.t().contiguous()

    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    has_bias = bias is not None
    # Triton always needs a pointer, even if unused when HAS_BIAS=False
    bias_ptr = bias if has_bias else out

    fused_linear_lse_leaky2_gelu2_kernel[grid](
        x, b, bias_ptr, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0),
        negative_slope,
        HAS_BIAS=has_bias,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated model:
      - Linear: x @ W^T + b
      - LogSumExp over dim=1 (keepdim=True)
      - LeakyReLU -> LeakyReLU
      - GELU (logistic approx) -> GELU (logistic approx)

    Entire pipeline is fused into a single Triton kernel with
    exactly one global store for the final [M, 1] output.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Match nn.Linear initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self.negative_slope = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be a CUDA tensor for Triton kernels."
        if x.dtype != torch.float32:
            x = x.float()

        y = fused_linear_lse_leaky2_gelu2(
            x, self.weight, self.bias, negative_slope=self.negative_slope
        )
        return y
