# Optimized Triton implementation replacing Gemm + Multiply + LeakyReLU

import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_gemm_mul_leakyrelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for each tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled matmul: A[M, K] @ B[K, N]
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias: bias is [N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Fused multiply by scalar
    acc = acc * multiplier

    # Fused LeakyReLU: x if x >= 0 else negative_slope * x
    # Implemented as: max(x, 0) + negative_slope * min(x, 0)
    zero = 0.0
    pos = tl.maximum(acc, zero)
    neg = tl.minimum(acc, zero) * negative_slope
    acc = pos + neg

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_gemm_mul_leakyrelu(x, weight, bias, multiplier, negative_slope):
    """
    x: [M, K]
    weight: [N, K] (nn.Linear weight, out_features x in_features)
    bias: [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"

    M, K = x.shape
    N = weight.shape[0]

    # Prepare output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # B = W^T to get [K, N] for A[M,K] @ B[K,N]
    b = weight.t().contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    fused_gemm_mul_leakyrelu_kernel[grid](
        x, b, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        float(multiplier), float(negative_slope),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
      Linear -> multiply by scalar -> LeakyReLU
    """

    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # Match nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        # Expect x to be on CUDA device
        return fused_gemm_mul_leakyrelu(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )


import math
