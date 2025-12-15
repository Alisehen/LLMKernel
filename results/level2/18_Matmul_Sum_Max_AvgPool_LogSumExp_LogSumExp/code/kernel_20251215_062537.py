# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_K": 64,
                "num_warps": 4,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_K": 64,
                "num_warps": 4,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_K": 128,
                "num_warps": 4,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_K": 128,
                "num_warps": 8,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_K": 64,
                "num_warps": 8,
            },
            num_stages=2,
        ),
    ],
    key=["M", "K"],
)
@triton.jit
def linear_row_sum_kernel(
    x_ptr,        # [M, K]
    wsum_ptr,     # [K]   = weight.sum(dim=0)
    b_sum,        # scalar = bias.sum()
    out_ptr,      # [M]
    M, K,
    stride_xm, stride_xk,
    stride_wk,
    stride_outm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute, for each row i:

        out[i] = sum_j (x @ W^T + b)[i, j]

    Using the identity:

        sum_j (x @ W^T + b)[i, j] =
            x[i, :] · (sum_j W[j, :]) + sum_j b[j]

    So the kernel only performs:

        out[i] = x[i, :] · w_sum + b_sum

    where w_sum = weight.sum(dim=0), b_sum = bias.sum().

    Accumulation is done in fp32; result is written in the
    dtype of out_ptr (implicit cast on store).

    Memory-fusion constraint:
    - Multiple input loads (x, w_sum) are allowed.
    - Exactly one store to out_ptr; no intermediate stores.
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for the initial K-tile
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = wsum_ptr + offs_k * stride_wk

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        x = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask,
            other=0.0,
        )

        # Use tensor-core friendly dot for fp16/bf16 inputs.
        # Inputs remain in their native dtype; accumulation in fp32.
        # Shape: [BLOCK_M, BLOCK_K] @ [BLOCK_K, 1] -> [BLOCK_M, 1]
        acc += tl.dot(x, w[:, None], out_dtype=tl.float32)[:, 0]

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    # Add scalar bias sum in fp32
    acc += b_sum

    out_ptrs = out_ptr + offs_m * stride_outm
    # Single final store; all intermediate values stay in registers
    tl.store(out_ptrs, acc, mask=mask_m)


def fused_linear_with_reductions(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused replacement for the reference sequence:

        y = F.linear(x, weight, bias)          # (M, N)
        y = torch.sum(y, dim=1, keepdim=True)  # (M, 1)
        y = torch.max(y, dim=1, keepdim=True)[0]
        y = torch.mean(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)

    For a single element per row after the first sum, all subsequent
    reductions are mathematical no-ops, so the whole chain reduces to:

        y = F.linear(x, weight, bias).sum(dim=1, keepdim=True)

    For float32 inputs, we rely on PyTorch/cuBLAS for strong numerical
    behavior. For lower-precision dtypes (float16/bfloat16), we use a
    high-performance Triton GEMV-style kernel that exploits:

        sum_j (x @ W^T + b)[i, j] =
            x[i, :] · (sum_j W[j, :]) + sum_j b[j]
    """
    assert x.dim() == 2, "x must be 2D (batch_size, in_features)"
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "in_features mismatch between x and weight"
    assert bias.shape == (N,), "bias must be 1D of shape (out_features,)"

    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"

    # Float32 path: rely on highly optimized cuBLAS and exact-ish numerics.
    if x.dtype == torch.float32:
        y = torch.nn.functional.linear(x, weight, bias)
        y = y.sum(dim=1, keepdim=True)
        return y

    # Low-precision path: use Triton.
    # Precompute column-wise sum of weight and full sum of bias:
    #   w_sum[k] = sum_j W[j, k]
    #   b_sum    = sum_j bias[j]
    # These are computed once and then streamed efficiently inside the kernel.
    w_sum = weight.sum(dim=0).contiguous()
    b_sum = bias.sum(dtype=torch.float32)

    # Output is one scalar per row
    y_vec = torch.empty(M, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    linear_row_sum_kernel[grid](
        x,
        w_sum,
        b_sum,
        y_vec,
        M,
        K,
        x.stride(0),
        x.stride(1),
        w_sum.stride(0),
        y_vec.stride(0),
    )

    y = y_vec.view(M, 1)
    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Match reference model structure: nn.Linear submodule named "linear"
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the fused implementation; this will dispatch to Triton for
        # low-precision dtypes and to cuBLAS (via torch.nn.functional.linear)
        # for float32.
        return fused_linear_with_reductions(x, self.linear.weight, self.linear.bias)
