# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gemm_swish_scale_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] = weight^T
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scaling_factor,  # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D launch grid
    pid_m = tl.program_id(0)  # along M dimension
    pid_n = tl.program_id(1)  # along N dimension

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer arithmetic for tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias: [N] -> broadcast to [BLOCK_M, BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Swish activation: x * sigmoid(x), then scale
    # sigmoid(x) = 1 / (1 + exp(-x))
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sigmoid = 1.0 / (1.0 + exp_neg)
    acc = acc * sigmoid
    acc = acc * scaling_factor

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_swish_scale(x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: torch.Tensor,
                             scaling_factor: float) -> torch.Tensor:
    """
    x: [M, K]
    weight: [N, K] (same as nn.Linear.weight)
    bias: [N]
    scaling_factor: scalar float
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "This implementation assumes float32"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "Incompatible shapes for matmul"

    # Triton kernel expects B = weight^T with shape [K, N]
    b = weight.t().contiguous()

    # Allocate output
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_gemm_swish_scale_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scaling_factor,
    )
    return c


class ModelNew(nn.Module):
    """
    Replacement for the original PyTorch Model using a fused Triton kernel:
    Linear (matmul + bias) + Swish activation + scaling.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = float(scaling_factor)
        # Initialize similar to nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x to be on CUDA for Triton
        return fused_linear_swish_scale(x, self.weight, self.bias, self.scaling_factor)


import math  # placed at end to comply with "import all modules you use"
