import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused Triton kernel:
#   y = x @ W^T + b          (GEMM + bias)
#   y_pooled = MaxPool1d(kernel_size=2, stride=2) along feature dim
#   out[b] = scale * sum(y_pooled[b, :])
#
# This kernel avoids materializing the [M, N] intermediate and directly
# produces [M] outputs, drastically reducing DRAM traffic. Designed for Ada.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Conservative baseline (required): good for general workloads
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N2": 64,  # N2 = N // 2 (pooled positions)
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Larger M tile, narrower N2: better when M is large and N moderate
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N2": 32,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Narrower M tile, wider N2: better when N is large
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N2": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N2", "K"],
)
@triton.jit
def linear_maxpool_sum_scale_kernel(
    a_ptr,        # [M, K]              input
    b_ptr,        # [K, N]              weight.T (contiguous in N)
    bias_ptr,     # [N]
    out_ptr,      # [M]                 fp32 partial sums (atomic adds)
    M, N, N2, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_out,
    scale,        # float32
    BLOCK_M: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program id and 2D tiling over (M, N2)
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n2 = tl.cdiv(N2, BLOCK_N2)

    num_pid_in_group = GROUP_M * num_pid_n2
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n2 = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n2 = offs_n2 < N2

    # Map pooled positions to original N-dim columns: (2*i, 2*i+1)
    cols0 = offs_n2 * 2
    cols1 = cols0 + 1
    mask_cols0 = cols0 < N
    mask_cols1 = cols1 < N

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n2, BLOCK_N2)
    tl.multiple_of(offs_k, BLOCK_K)

    # Accumulators for two adjacent columns (before max-pooling)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

    # Base pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b0_ptrs = b_ptr + offs_k[:, None] * stride_bk + cols0[None, :] * stride_bn
    b1_ptrs = b_ptr + offs_k[:, None] * stride_bk + cols1[None, :] * stride_bn

    # K-loop with small BLOCK_K to keep register pressure manageable
    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        k_mask = offs_k < k_rem

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b0 = tl.load(
            b0_ptrs,
            mask=k_mask[:, None] & mask_cols0[None, :],
            other=0.0,
        )
        b1 = tl.load(
            b1_ptrs,
            mask=k_mask[:, None] & mask_cols1[None, :],
            other=0.0,
        )

        acc0 += tl.dot(a, b0, allow_tf32=True)
        acc1 += tl.dot(a, b1, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b0_ptrs += BLOCK_K * stride_bk
        b1_ptrs += BLOCK_K * stride_bk

    # Load bias for both columns in each window and add
    bias0 = tl.load(
        bias_ptr + cols0 * stride_bias,
        mask=mask_cols0,
        other=0.0,
    ).to(tl.float32)
    bias1 = tl.load(
        bias_ptr + cols1 * stride_bias,
        mask=mask_cols1,
        other=0.0,
    ).to(tl.float32)

    acc0 += bias0[None, :]
    acc1 += bias1[None, :]

    # Max over the 2-element pooling window
    pair_max = tl.maximum(acc0, acc1)

    # Sum over pooled feature positions (N2 dim)
    row_sum = tl.sum(pair_max, axis=1)  # [BLOCK_M]

    # Scale and atomically accumulate into output
    row_sum_scaled = row_sum * scale
    out_ptrs = out_ptr + offs_m * stride_out
    tl.atomic_add(out_ptrs, row_sum_scaled, mask=mask_m)


# ---------------------------------------------------------------------------
# Wrapper: fused linear + maxpool + sum + scale
# ---------------------------------------------------------------------------

def triton_linear_maxpool_sum_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K] (same as nn.Linear.weight)
    bias:   [N]
    Returns:
      out: [M], where
        y = x @ weight.T + bias
        y_pooled = MaxPool1d(kernel_size=2, stride=2) along last dim of y
        out[m] = scale * sum(y_pooled[m, :])
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "in_features mismatch"
    assert bias.shape[0] == N
    assert N >= 2, "Need at least 2 output features for pooling"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype"

    # Use weight.T as [K, N] contiguous in N for GEMM-friendly layout
    b = weight.t().contiguous()  # [K, N]

    N2 = N // 2  # number of pooled positions, floor as in MaxPool1d(kernel_size=2, stride=2)
    assert N2 > 0, "Output feature dim too small for pooling"

    # Accumulate into fp32 buffer to simplify atomics and ensure numeric stability
    out_fp32 = torch.zeros((M,), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"])
            * triton.cdiv(N2, meta["BLOCK_N2"]),
        )

    linear_maxpool_sum_scale_kernel[grid](
        x,
        b,
        bias,
        out_fp32,
        M,
        N,
        N2,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0),
        out_fp32.stride(0),
        float(scale),
    )

    # Match original API: return in the same dtype as input
    return out_fp32.to(x.dtype)


# ---------------------------------------------------------------------------
# PyTorch module using fused Triton kernel
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Model:
      x -> linear (x @ W^T + b) -> MaxPool1d(kernel_size=2, stride=2) along features
        -> sum over pooled features -> scale
    Implemented as a single fused Triton kernel for maximal performance.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        assert kernel_size == 2, "This implementation assumes kernel_size=2"
        self.in_features = in_features
        self.out_features = out_features
        self.scale_factor = float(scale_factor)

        # Parameters analogous to nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input must be a CUDA tensor for Triton kernels"
        # Fused: linear + maxpool + sum + scale
        return triton_linear_maxpool_sum_scale(x, self.weight, self.bias, self.scale_factor)
