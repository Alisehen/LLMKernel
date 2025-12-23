import torch, torch.nn as nn, triton, triton.language as tl


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
    # Program ids for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply accumulate
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

        # Advance pointers to next K-tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcast over M)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Swish activation: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sig = 1.0 / (1.0 + exp_neg)
    acc = acc * sig

    # Scale by scalar factor
    acc = acc * scaling_factor

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear_swish_scale(x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: torch.Tensor,
                             scaling_factor: float) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K] (same layout as nn.Linear.weight)
    bias:   [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    M, K = x.shape
    N = weight.shape[0]
    # Ensure dimensions agree
    assert weight.shape[1] == K
    assert bias.shape[0] == N

    # Convert weight to [K, N] for GEMM (A: [M,K], B: [K,N])
    b = weight.t().contiguous()

    # Output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Grid: 2D over M and N
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"])
        )

    fused_linear_swish_scale_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scaling_factor,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=8,
        num_stages=4,
    )
    return c


class ModelNew(nn.Module):
    """
    Fused implementation of:
      y = (x @ W^T + b)
      y = y * sigmoid(y)   # Swish
      y = y * scaling_factor
    using a high-performance Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Match nn.Linear parameter shapes: weight [out_features, in_features], bias [out_features]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_swish_scale(x, self.weight, self.bias, self.scaling_factor)
