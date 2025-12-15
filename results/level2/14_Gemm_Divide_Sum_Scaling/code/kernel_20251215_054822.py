# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_div_sum_scale_kernel(
    x_ptr,         # *dtype, [M, K]
    w_ptr,         # *dtype, [N, K]  (weight, NOT pre-summed, N = hidden_size)
    y_ptr,         # *float32, [M]
    M, N, K,       # int32
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym,
    alpha,         # float32 scalar = scaling_factor / 2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel computing:

        y[m] = alpha * sum_n ( (x @ w.T)[m, n] )

    where:
        alpha = scaling_factor / 2

    This preserves the original algorithm:
        out = x @ weight.T
        out = out / 2
        y = scaling_factor * out.sum(dim=-1)
    without incorrectly collapsing weight along K beforehand.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator for this (M-tile, N-tile) block of the matmul result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k
        mask_k = k_ids < K

        # Load A tile: X[offs_m, k_ids]  -> shape (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + (
            offs_m[:, None] * stride_xm + k_ids[None, :] * stride_xk
        )
        x = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load B tile as (K, N-block) by viewing W as B = W^T:
        #   B[k, n] == W[n, k]
        # So pointers: W[offs_n, k_ids] but arranged as (k, n)
        w_ptrs = w_ptr + (
            k_ids[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        x = x.to(tl.float32)
        w = w.to(tl.float32)

        # Matrix multiply for this tile: (BM, BK) x (BK, BN) -> (BM, BN)
        acc += tl.dot(x, w)

    # For this tile, we now have:
    #   acc[m, n_in_tile] == (x @ w.T)[m, n]
    # We want partial sum over n in this tile, times alpha
    partial = tl.sum(acc, axis=1) * alpha  # shape (BLOCK_M,)

    # Accumulate into y[m] across N-tiles via atomic add
    y_ptrs = y_ptr + offs_m * stride_ym
    tl.atomic_add(y_ptrs, partial, mask=mask_m)


def fused_matmul_div_sum_scale(x: torch.Tensor, weight: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Compute, on CUDA with Triton:

        out = x @ weight.T           # [M, H]
        out = out / 2
        y   = scaling_factor * out.sum(dim=-1, keepdim=True)    # [M, 1]

    This keeps the original matmul → divide → sum algorithm intact,
    avoiding the incorrect simplification to x @ weight.sum(dim=0).
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
    assert x.dtype == weight.dtype, "x and weight must have the same dtype"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Supported dtypes: fp16, bf16, fp32"

    M, K = x.shape
    H, K_w = weight.shape
    assert K_w == K, "Incompatible shapes between x and weight"
    N = H  # rename for kernel

    # Combined scaling: divide by 2, then multiply by scaling_factor
    alpha = float(scaling_factor) * 0.5

    # Accumulate in float32 for numeric stability and atomic support
    y_fp32 = torch.zeros((M,), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_div_sum_scale_kernel[grid](
        x,
        weight,
        y_fp32,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y_fp32.stride(0),
        alpha,
    )

    # Cast back to original dtype if needed and add singleton dim
    if x.dtype != torch.float32:
        y = y_fp32.to(x.dtype)
    else:
        y = y_fp32

    return y.view(M, 1)


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original model.

    Implements:
        out = x @ weight.T
        out = out / 2
        y   = scaling_factor * out.sum(dim=-1, keepdim=True)
    using a fused Triton kernel that performs matmul and reduction
    without changing the underlying algorithm.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)
