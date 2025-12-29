import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_linear_relu_div_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    inv_div,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tile of output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-blocks of A and B for this tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for numeric stability / performance
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Masks for bounds
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # FMA via tensor core-friendly dot
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Divide by constant via multiplication with precomputed inverse
    acc = acc * inv_div

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear_relu_div(x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: torch.Tensor,
                          divisor: float) -> torch.Tensor:
    """
    Fused implementation of:
        y = relu(x @ weight.T + bias) / divisor
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    M, K = x.shape
    out_features, in_features = weight.shape
    assert in_features == K
    N = out_features

    # Prepare weight as (K, N) for GEMM
    b = weight.t().contiguous()

    # Output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Grid: one program per BLOCK_M x BLOCK_N tile of C
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    inv_div = 1.0 / float(divisor)

    fused_linear_relu_div_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        inv_div,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=8, num_stages=3,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
        y = relu(Linear(x)) / divisor
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.divisor = float(divisor)

    def forward(self, x):
        # Expect x on CUDA; the benchmarking harness should move it if needed.
        return fused_linear_relu_div(x, self.weight, self.bias, self.divisor)
