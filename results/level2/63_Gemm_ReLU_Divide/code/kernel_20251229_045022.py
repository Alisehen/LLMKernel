# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Compute-tilted, large tiles – good for big, square-ish problems
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        # Baseline, conservative config (required): num_warps=4, num_stages=2
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        ),
        # Smaller rectangular tile for high-register-pressure / skinny shapes
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_relu_div_kernel_mixed(
    a_ptr,  # [M, K]  (x, mixed precision)
    b_ptr,  # [K, N]  (weight.T, mixed precision)
    bias_ptr,  # [N]   (fp32)
    c_ptr,  # [M, N]  (fp32 output)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    inv_div,  # scalar fp32 = 1.0 / divisor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ------------------------------------------------------------------
    # 2D program id -> tile coordinates
    # ------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment / divisibility hints help the compiler generate better code
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Output tile pointer
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # ------------------------------------------------------------------
    # GEMM: A[M, K] @ B[K, N] -> C[M, N] (acc in fp32)
    # ------------------------------------------------------------------
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    # Loop over K dimension in BLOCK_K chunks
    while k < K:
        k_idx = k + offs_k
        k_mask = k_idx < K

        # Masks for in-bounds loads
        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        # Coalesced loads for A and B tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Mixed-precision matmul on Tensor Cores, accumulate in fp32
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        # Advance pointers for next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # ------------------------------------------------------------------
    # Fused epilogue: bias + ReLU + div in fp32
    # ------------------------------------------------------------------
    # Bias is 1D over N, broadcast over M
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # ReLU in-place
    acc = tl.maximum(acc, 0.0)

    # Divide by constant via precomputed inverse (cheaper than true div)
    acc *= inv_div

    # Single final store – no intermediates ever written to memory
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear_relu_div(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
) -> torch.Tensor:
    """
    Fused implementation of:
        y = relu(x @ weight.T + bias) / divisor

    x      : [M, K] (fp16 / bf16 / fp32)
    weight : [N, K] (fp16 / bf16, stored as [out_features, in_features])
    bias   : [N]    (fp32)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert bias.dtype == torch.float32
    assert weight.dtype in (torch.float16, torch.bfloat16)
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)

    M, K = x.shape
    out_features, in_features = weight.shape
    assert in_features == K
    N = out_features

    # Cast x to weight dtype for Tensor Cores
    if x.dtype != weight.dtype:
        x_mixed = x.to(weight.dtype)
    else:
        x_mixed = x

    # Ensure weight contiguous, then view as [K, N]
    if not weight.is_contiguous():
        weight_contig = weight.contiguous()
    else:
        weight_contig = weight
    b = weight_contig.t()  # [K, N]

    # Output in fp32
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    inv_div = 1.0 / float(divisor)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_relu_div_kernel_mixed[grid](
        x_mixed,
        b,
        bias,
        c,
        M,
        N,
        K,
        x_mixed.stride(0),
        x_mixed.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        inv_div,
    )

    return c


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
        y = relu(Linear(x)) / divisor
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.float16)
        )
        self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.float32))
        self.divisor = float(divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_relu_div(x, self.weight, self.bias, self.divisor)
