import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_swish_add_two_biases_kernel(
    a_ptr,          # [M, K] input
    b_ptr,          # [K, N] weight^T
    linear_bias_ptr,  # [N] bias of the linear layer (pre-activation)
    extra_bias_ptr,   # [N] extra bias added after Swish
    c_ptr,          # [M, N] output
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = Swish(A @ B + linear_bias) + extra_bias,
    where Swish(x) = x * sigmoid(x).

    Shapes:
      A: [M, K]
      B: [K, N]
      linear_bias: [N]
      extra_bias: [N]
      C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul: A[M,K] @ B[K,N] -> acc[M,N]
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add linear (pre-activation) bias: broadcast along M
    linear_bias = tl.load(
        linear_bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    )
    x = acc + linear_bias[None, :]

    # Swish activation: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-x))
    x = x * sig

    # Add extra bias after Swish: broadcast along M
    extra_bias = tl.load(
        extra_bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    )
    x = x + extra_bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, x, mask=c_mask)


def matmul_swish_add_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    linear_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    """
    x:           [M, K]
    weight:      [N, K]  (nn.Linear weight: [out_features, in_features])
    linear_bias: [N]     (nn.Linear bias: pre-activation)
    extra_bias:  [N]     (additional bias applied after Swish)
    returns:     [M, N]

    Computes: Swish(x @ weight.T + linear_bias) + extra_bias
    """
    assert x.dim() == 2, "Input x must be 2D (batch_size, in_features)"
    assert weight.dim() == 2, "Weight must be 2D (out_features, in_features)"
    assert linear_bias.dim() == 1, "linear_bias must be 1D (out_features,)"
    assert extra_bias.dim() == 1, "extra_bias must be 1D (out_features,)"

    M, K = x.shape
    out_features, in_features = weight.shape
    assert K == in_features, "Incompatible shapes for matmul"
    N = out_features
    assert linear_bias.shape[0] == N, "linear_bias size mismatch"
    assert extra_bias.shape[0] == N, "extra_bias size mismatch"

    x_contig = x.contiguous()
    # B has shape [K, N] for A[M,K] x B[K,N] -> C[M,N]
    b_t = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_swish_add_two_biases_kernel[grid](
        x_contig, b_t, linear_bias, extra_bias, c,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original model.

    Pipeline:
      1) Linear: x @ W^T + linear_bias
      2) Swish activation: x * sigmoid(x)
      3) Add extra bias term
      4) GroupNorm over features (channels)
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Weight for the linear layer (no built-in bias; we handle bias explicitly)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # Bias of the "logical" nn.Linear layer (pre-activation)
        self.linear_bias = nn.Parameter(torch.randn(out_features))
        # Extra bias added after Swish
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Same GroupNorm as in the original model
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_features]
        returns: [batch_size, out_features]
        """
        x = matmul_swish_add_bias(x, self.weight, self.linear_bias, self.bias)
        x = self.group_norm(x)
        return x
