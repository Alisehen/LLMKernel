# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # High-throughput, still conservative on register pressure
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Rectangular tiles for tall/skinny or wide shapes
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Small fallback – lowest register footprint, highest occupancy
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_div_gelu_kernel(
    a_ptr, w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    inv_divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling of output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Pointers for A: [M, K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Pointers for W (weight): [N, K], accessed as B[K, N] with B(k, n) = W(n, k)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator in FP32 to leverage tensor cores (TF32) for fp16/bf16/fp32 inputs
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K

        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # Use tensor cores (TF32) where beneficial
        acc += tl.dot(a, w, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Fused elementwise ops
    # Bias: [N] broadcast along M
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Divide by scalar via multiplication by its inverse
    acc *= inv_divisor

    # GELU using erf formulation:
    # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Keep only one temporary to control register pressure
    t = tl.math.erf(acc * 0.7071067811865476)  # 1 / sqrt(2)
    acc = acc * (1.0 + t) * 0.5

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def fused_linear_div_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Fused implementation of:
        y = x @ weight.T + bias
        y = y / divisor
        y = GELU(y)

    Shapes:
        x:      [M, K]
        weight: [N, K]  (same as nn.Linear.weight)
        bias:   [N]
        out:    [M, N]
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda and bias.is_cuda, "Parameters must be on CUDA device"
    assert x.dtype == weight.dtype == bias.dtype, "dtypes of x, weight, bias must match"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "weight shape must be [N, K]"
    assert bias.shape[0] == N, "bias shape must match output features"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    inv_divisor = 1.0 / float(divisor)

    linear_div_gelu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        # Map W [N, K] as B [K, N]:
        # B(k, n) = W(n, k) -> stride_wk = stride over K, stride_wn = stride over N
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        inv_divisor,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
        y = Linear(x)
        y = y / divisor
        y = GELU(y)
    """

    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.divisor = float(divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
            self.weight.data = self.weight.data.cuda()
            self.bias.data = self.bias.data.cuda()
        return fused_linear_div_gelu(x, self.weight, self.bias, self.divisor)
