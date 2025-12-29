import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_swish_scale_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Swish: x * sigmoid(x), with manual sigmoid
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig * scaling_factor

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear_swish_scale(x, weight, bias, scaling_factor: float):
    """
    x:       [M, K]
    weight:  [N, K]  (same as nn.Linear.weight)
    bias:    [N]
    output:  [M, N] = Swish(x @ weight.T + bias) * scaling_factor
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype
    M, K = x.shape
    N, K_w = weight.shape
    assert K_w == K, "Weight shape must be [N, K] with K == in_features"

    # Convert to GEMM form: [M, K] @ [K, N]
    b = weight.t().contiguous()  # [K, N]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_swish_scale_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scaling_factor,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=2,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
        y = Linear(x)
        y = y * sigmoid(y)   # Swish
        y = y * scaling_factor
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = float(scaling_factor)

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_swish_scale(x, self.weight, self.bias, self.scaling_factor)
