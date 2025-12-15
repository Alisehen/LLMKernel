# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Main high-throughput config: balanced tile, conservative stages
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Rectangular tiles to adapt to tall / wide matrices
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_swish_add_two_biases_kernel(
    a_ptr,            # [M, K]
    b_ptr,            # [K, N]
    linear_bias_ptr,  # [N]
    extra_bias_ptr,   # [N]
    c_ptr,            # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = Swish(A @ B + linear_bias) + extra_bias
    where Swish(x) = x / (1 + exp(-x)).
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for precision and Tensor Core/TF32 usage
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    k = 0
    while k < K:
        k_next = k + BLOCK_K

        # Recompute cheap masks to reduce register pressure
        a_mask = (offs_m[:, None] < M) & (tl.arange(0, BLOCK_K)[None, :] + k < K)
        b_mask = (tl.arange(0, BLOCK_K)[:, None] + k < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k = k_next

    # Biases: recompute column mask instead of storing it
    linear_bias = tl.load(
        linear_bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    ).to(acc.dtype)

    extra_bias = tl.load(
        extra_bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
    ).to(acc.dtype)

    # Fused: x = acc + linear_bias; Swish(x) = x / (1 + exp(-x)); + extra_bias
    acc = acc + linear_bias[None, :]
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    acc = acc / (1.0 + exp_neg)
    acc = acc + extra_bias[None, :]

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=out_mask)


def matmul_swish_add_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    linear_bias: torch.Tensor,
    extra_bias: torch.Tensor,
) -> torch.Tensor:
    """
    x:           [M, K]
    weight:      [N, K]  (nn.Linear weight: [out_features, in_features])
    linear_bias: [N]
    extra_bias:  [N]
    returns:     [M, N]

    Computes: Swish(x @ weight.T + linear_bias) + extra_bias
    where Swish(z) = z / (1 + exp(-z)).
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

    # Ensure layouts: A [M,K] row-major, B [K,N] row-major
    x_contig = x.contiguous()
    b_t = weight.t().contiguous()  # [K, N]

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_swish_add_two_biases_kernel[grid](
        x_contig, b_t,
        linear_bias, extra_bias,
        c,
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
      2) Swish activation: x / (1 + exp(-x))
      3) Add extra bias term
      4) GroupNorm over features (channels)
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, in_features]
        returns: [batch_size, out_features]
        """
        x = matmul_swish_add_bias(
            x,
            self.linear.weight,
            self.linear.bias,
            self.bias,
        )
        x = self.group_norm(x)
        return x
