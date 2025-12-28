import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_row_sum_kernel(
    a_ptr,        # (M, K) input
    w_ptr,        # (K, N) weight^T (transposed weight)
    b_ptr,        # (N,) bias
    out_ptr,      # (M,) output row-sum accumulator
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        # Masks for valid indices
        m_mask = offs_m[:, None] < M            # (BM, 1)
        n_mask = offs_n[None, :] < N            # (1, BN)
        k_mask_a = offs_k[None, :] < k_remaining  # (1, BK) for A
        k_mask_w = offs_k[:, None] < k_remaining  # (BK, 1) for W

        # Load tiles with proper masking
        a = tl.load(a_ptrs, mask=m_mask & k_mask_a, other=0.0)      # (BM, BK)
        w = tl.load(w_ptrs, mask=k_mask_w & n_mask, other=0.0)      # (BK, BN)

        # Fused matmul in fp32 accumulator
        acc += tl.dot(a, w, allow_tf32=True)

        # Advance K-tile pointers
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        k_remaining -= BLOCK_K

    # Add bias along N dimension
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)  # (BN,)
    acc += bias[None, :]  # broadcast over M

    # Row-wise sum over features (N)
    row_sum = tl.sum(acc, axis=1)  # (BM,)

    # Accumulate into global output using atomics over M
    mask_m = offs_m < M
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


def fused_linear_and_reductions(x, weight, bias):
    """
    Computes:
        y = x @ weight.T + bias
        s = torch.sum(y, dim=1, keepdim=True)
        s = torch.max(s, dim=1, keepdim=True)[0]
        s = torch.mean(s, dim=1, keepdim=True)
        s = torch.logsumexp(s, dim=1, keepdim=True)
        s = torch.logsumexp(s, dim=1, keepdim=True)

    For the given model, all reductions after the first sum are no-ops
    because dim=1 has size 1, so the final result equals the row-wise sum.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Use weight^T with (K, N) layout for efficient access in Triton
    w_t = weight.t().contiguous()

    # Accumulator buffer for row sums
    out_fp32 = torch.zeros((M,), device=x.device, dtype=torch.float32)

    # 2D grid over M (rows) and N (columns)
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    fused_linear_row_sum_kernel[grid](
        x, w_t, bias, out_fp32,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )

    out = out_fp32.to(x.dtype).view(M, 1)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel:
        - Linear (matmul + bias)
        - Sum over features
        - Max / Mean / LogSumExp / LogSumExp over size-1 dimension (no-ops)
    Final output matches the original PyTorch model.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_and_reductions(x, self.weight, self.bias)
