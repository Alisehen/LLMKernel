# complete ModelNew code with optimized Triton kernels
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_forward_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    linear_forward_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


@triton.jit
def batchnorm_reduce_kernel(
    x_ptr, mean_ptr, inv_std_ptr,
    M, N,
    stride_xm, stride_xn,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    sum_val = tl.zeros((BLOCK_N,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        mask = mask_m[:, None] & mask_n[None, :]

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    m_float = tl.full((1,), M, dtype=tl.float32)
    mean = sum_val / m_float
    var = sum_sq / m_float - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + offs_n, mean, mask=mask_n)
    tl.store(inv_std_ptr + offs_n, inv_std, mask=mask_n)


def triton_batchnorm_reduce(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    eps: float,
):
    M, N = x.shape
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    batchnorm_reduce_kernel[grid](
        x, mean, inv_std,
        M, N,
        x.stride(0), x.stride(1),
        eps,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=4,
    )


@triton.jit
def batchnorm_act_kernel(
    x_ptr, mean_ptr, inv_std_ptr, weight_ptr, bias_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    inv_std = tl.load(inv_std_ptr + offs_n, mask=offs_n < N, other=0.0)
    gamma = tl.load(weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    beta = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    mean = mean[None, :]
    inv_std = inv_std[None, :]
    gamma = gamma[None, :]
    beta = beta[None, :]

    y = (x - mean) * inv_std
    y = y * gamma + beta

    # GELU (tanh approximation)
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    y_cubed = y * y * y
    inner = k0 * (y + k1 * y_cubed)
    exp_2inner = tl.exp(2.0 * inner)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    y = 0.5 * y * (1.0 + tanh_inner)

    # ReLU
    y = tl.maximum(y, 0.0)

    tl.store(y_ptrs, y, mask=mask)


def triton_batchnorm_act(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    batchnorm_act_kernel[grid](
        x, mean, inv_std, weight, bias, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    """
    Fused Triton implementation of:
      Linear (GEMM + bias) -> BatchNorm1d (training-mode stats) -> GELU -> ReLU.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # BatchNorm1d affine parameters (gamma, beta)
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_eps = 1e-5

        # Initialize linear as nn.Linear does
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x to be on the same device as parameters (typically CUDA)
        x = x.to(self.weight.dtype)

        # Linear GEMM + bias via Triton
        y = triton_linear(x, self.weight, self.bias)

        M, N = y.shape
        mean = torch.empty(N, device=y.device, dtype=torch.float32)
        inv_std = torch.empty_like(mean)

        # BatchNorm statistics (training-style, per-batch)
        triton_batchnorm_reduce(y, mean, inv_std, self.bn_eps)

        # BatchNorm affine + GELU + ReLU
        out = triton_batchnorm_act(y, mean, inv_std, self.bn_weight, self.bn_bias)
        return out
