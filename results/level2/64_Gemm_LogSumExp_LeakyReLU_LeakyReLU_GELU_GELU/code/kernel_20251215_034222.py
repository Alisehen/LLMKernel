# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# -----------------------------
# GEMM + Bias kernel (Linear)
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k + k < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Write C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# -----------------------------
# Row-wise LogSumExp + 2x LeakyReLU + 2x GELU kernel
# -----------------------------
@triton.jit
def _gelu_logistic_approx(x: tl.tensor) -> tl.tensor:
    # GELU(x) ≈ x * Φ(x) with Φ logistic CDF: 1 / (1 + exp(-k*x))
    k = 1.702
    return x / (1.0 + tl.exp(-k * x))


@triton.jit
def rowwise_lse_leaky2_gelu2_kernel(
    x_ptr,      # [M, N]
    out_ptr,    # [M, 1]
    M, N,
    stride_xm, stride_xn,
    stride_om,
    negative_slope,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    row_offset = pid * stride_xm

    # Numerically stable log-sum-exp over the row
    m = -float("inf")
    s = 0.0

    for start_n in range(0, N, BLOCK_N):
        offs = start_n + tl.arange(0, BLOCK_N)
        mask = offs < N

        ptrs = x_ptr + row_offset + offs * stride_xn
        x = tl.load(ptrs, mask=mask, other=-float("inf"))

        tile_max = tl.max(x, axis=0)
        x_rel = x - tile_max
        tile_exp = tl.exp(x_rel)
        tile_sum = tl.sum(tile_exp, axis=0)

        m_new = tl.maximum(m, tile_max)
        coef_old = tl.exp(m - m_new)
        coef_tile = tl.exp(tile_max - m_new)
        s = s * coef_old + tile_sum * coef_tile
        m = m_new

    lse = m + tl.log(s)

    # Two LeakyReLU activations
    neg = negative_slope
    y = tl.where(lse >= 0, lse, neg * lse)
    y = tl.where(y >= 0, y, neg * y)

    # Two GELU (approximate) activations
    y = _gelu_logistic_approx(y)
    y = _gelu_logistic_approx(y)

    # Store result [M, 1]
    out_row_ptr = out_ptr + pid * stride_om
    tl.store(out_row_ptr, y)


# -----------------------------
# Wrapper functions
# -----------------------------
def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [N, K] (nn.Linear.weight)
    bias: [N]
    Returns: [M, N] = x @ weight^T + bias
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32
    assert weight.dtype == torch.float32
    assert bias.dtype == torch.float32

    M, K = x.shape
    N = weight.shape[0]

    # B matrix: [K, N]
    b = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_gemm_bias_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def fused_rowwise_lse_leaky2_gelu2(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    x: [M, N]
    Returns: [M, 1]
    """
    assert x.is_cuda
    assert x.dtype == torch.float32

    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    # Choose BLOCK_N up to 1024, power-of-two for efficient reductions
    if N >= 1024:
        BLOCK_N = 1024
    else:
        pow2 = 1 << (N - 1).bit_length()
        BLOCK_N = min(max(pow2, 64), 1024)

    grid = (M,)

    rowwise_lse_leaky2_gelu2_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0),
        negative_slope,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


# -----------------------------
# Torch Module
# -----------------------------
class ModelNew(nn.Module):
    """
    Triton-accelerated model:
    - Linear (GEMM + bias)
    - Row-wise LogSumExp (dim=-1, keepdim=True)
    - LeakyReLU -> LeakyReLU
    - GELU (approx) -> GELU (approx)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Use a Python float for 'a' to satisfy kaiming_uniform_ API
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self.negative_slope = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be a CUDA tensor for Triton kernels."
        if x.dtype != torch.float32:
            x = x.float()

        # Linear: x @ W^T + b
        y = fused_linear(x, self.weight, self.bias)

        # Row-wise LogSumExp + activations
        y = fused_rowwise_lse_leaky2_gelu2(y, negative_slope=self.negative_slope)
        return y
