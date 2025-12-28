# <complete ModelNew code with optimized Triton kernels>

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_sum_kernel(
    a_ptr,          # pointer to input matrix A, shape (M, K)
    w_ptr,          # pointer to reduced weight vector, shape (K,)
    bias,           # scalar bias after reduction
    out_ptr,        # pointer to output vector, shape (M,)
    M, K,           # sizes
    stride_am, stride_ak,
    stride_wk,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k in range(0, K, BLOCK_K):
        k_curr = k + offs_k
        mask_k = k_curr < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_curr[None, :] * stride_ak
        w_ptrs = w_ptr + k_curr * stride_wk

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc += tl.sum(a * w[None, :], axis=1)

    acc = acc + bias

    out_ptrs = out_ptr + offs_m * stride_om
    tl.store(out_ptrs, acc, mask=mask_m)


def fused_linear_chain(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of the original sequence:
        y = linear(x)
        y = sum(y, dim=1, keepdim=True)
        y = max(y, dim=1, keepdim=True)[0]
        y = mean(y, dim=1, keepdim=True)
        y = logsumexp(y, dim=1, keepdim=True)
        y = logsumexp(y, dim=1, keepdim=True)

    Mathematically simplifies to:
        y = x @ W_reduced + b_reduced
    where:
        W_reduced = weight.sum(dim=0)
        b_reduced = bias.sum()
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda

    M, K = x.shape  # (batch_size, in_features)
    # weight: (out_features, in_features)
    # reduce over out_features to get effective single-output weights
    w_reduced = weight.sum(dim=0).contiguous()  # shape (K,)
    b_reduced = float(bias.sum().item())        # scalar

    x_c = x.contiguous()
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_K = 128

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    matvec_sum_kernel[grid](
        x_c,
        w_reduced,
        b_reduced,
        out,
        M,
        K,
        x_c.stride(0),
        x_c.stride(1),
        w_reduced.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=4,
    )

    if out.dtype != x.dtype:
        out = out.to(x.dtype)

    return out.view(M, 1)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model.

    Original sequence:
        - Linear (matrix multiplication + bias)
        - Sum over features
        - Max over singleton dim
        - Mean over singleton dim
        - LogSumExp over singleton dim (twice)

    All post-linear ops reduce to an affine map:
        output = x @ W_reduced + b_reduced

    This class keeps the same learnable parameters as nn.Linear but
    uses a fused Triton kernel for the simplified computation when on CUDA.
    """

    def __init__(self, in_features: int, out_features: int):
        super(ModelNew, self).__init__()
        # Match nn.Linear parameterization: (out_features, in_features) and (out_features,)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return fused_linear_chain(x, self.weight, self.bias)
        else:
            # CPU / non-CUDA fallback: exact original PyTorch behavior
            x = torch.nn.functional.linear(x, self.weight, self.bias)
            x = torch.sum(x, dim=1, keepdim=True)
            x = torch.max(x, dim=1, keepdim=True)[0]
            x = torch.mean(x, dim=1, keepdim=True)
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.logsumexp(x, dim=1, keepdim=True)
            return x
