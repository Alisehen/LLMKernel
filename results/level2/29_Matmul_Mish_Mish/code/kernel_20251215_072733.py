# <corrected code>
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish(x):
    """
    Numerically stable Mish: x * tanh(softplus(x))
    softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    """
    abs_x = tl.abs(x)
    sp = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x, 0.0)
    t = tl.exp(-2.0 * sp)
    return x * (1.0 - t) / (1.0 + t)


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_double_mish_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -----------------------------
    # 2D program id for output tile
    # -----------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Row/col masks reused by all fused ops
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to A and B tiles
    # A is (M, K) in row-major
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is (N, K) in row-major; we access as (K, N)
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # Accumulator in fp32 for good numerics
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---------------
    # Main K reduction
    # ---------------
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining

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

        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        # Advance along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ------------
    # Bias + Mish²
    # ------------

    # Bias: shape (N,), broadcast along M
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Apply Mish twice (double Mish)
    acc = mish(acc)
    acc = mish(acc)

    # -----------
    # Store result
    # -----------
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_double_mish(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Compute: y = mish(mish(x @ weight.T + bias))

    x:      (M, K)
    weight: (N, K)  -- same layout as nn.Linear.weight
    bias:   (N,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape[1] == weight.shape[1], "in_features mismatch between x and weight"
    assert weight.shape[0] == bias.shape[0], "out_features mismatch between weight and bias"

    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_double_mish_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized equivalent of:

        y = linear(x)
        y = mish(y)
        y = mish(y)
    """
    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        # Match nn.Linear parameter shapes
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # Initialize like nn.Linear (Kaiming-uniform + bias init)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_double_mish(x, self.weight, self.bias)
