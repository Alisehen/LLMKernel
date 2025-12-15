# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 4,
            },
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 8,
            },
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
    a_ptr,       # [M, K]
    w_ptr,       # [N, K] (nn.Linear.weight)
    bias_ptr,    # [N]
    out_ptr,     # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute:
        out = x @ W^T + b
    where:
        x: [M, K]
        W: [N, K]
        out: [M, N]
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers for the first K-tile
    # A is [M, K] with strides (stride_am, stride_ak)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # W is [N, K], accessed as [K, N] via (stride_wk, stride_wn)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )

        acc += tl.dot(a, w)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    # Add bias: broadcast over rows
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]

    # Store full linear output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fused_linear_with_reductions(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused replacement for:

        y = F.linear(x, weight, bias)          # (M, N)
        y = torch.sum(y, dim=1, keepdim=True)  # (M, 1)
        y = torch.max(y, dim=1, keepdim=True)[0]
        y = torch.mean(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = torch.logsumexp(y, dim=1, keepdim=True)

    The Triton kernel computes the linear part y = x @ W^T + b.
    The exact reduction chain (including the nonlinear ops) is then
    applied with PyTorch to ensure bit-correct behavior.
    """
    assert x.dim() == 2, "x must be 2D (batch_size, in_features)"
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "in_features mismatch between x and weight"
    assert bias.shape == (N,), "bias must be 1D of shape (out_features,)"

    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"

    # Allocate full linear output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
        x,
        weight,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
    )

    # Apply the full reduction chain in PyTorch to match the reference exactly
    y = torch.sum(y, dim=1, keepdim=True)
    y, _ = torch.max(y, dim=1, keepdim=True)
    y = torch.mean(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)
    y = torch.logsumexp(y, dim=1, keepdim=True)

    return y


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Match reference model structure: nn.Linear submodule named "linear"
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the Triton-accelerated linear layer and apply the same reductions
        return fused_linear_with_reductions(x, self.linear.weight, self.linear.bias)
