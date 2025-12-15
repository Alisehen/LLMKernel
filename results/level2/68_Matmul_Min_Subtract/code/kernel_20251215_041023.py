# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Well-balanced 128x128 tile, good for large GEMMs on Ada
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=8,
            num_stages=3,
        ),
        # More rectangular tiles for tall-skinny / short-wide cases
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_min_sub_kernel(
    a_ptr,       # [M, K] input, row-major
    b_ptr,       # [N, K] weight, row-major (same layout as nn.Linear.weight)
    bias_ptr,    # [N]
    c_ptr,       # [M, N] output, row-major
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    constant,    # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 1D program id with GROUP_M swizzling for better L2 reuse of weights
    # -------------------------------------------------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    # Tile offsets along M and N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Shared boundary masks for all fused ops (grid covers output [M, N])
    m_mask = offs_m < M
    n_mask = offs_n < N

    # -------------------------------------------------------------------------
    # Pointers for A and B tiles
    #   A: [M, K], row-major:  a[m, k]  -> stride_am, stride_ak
    #   B: [N, K], row-major, but reinterpreted as [K, N] via strides:
    #      b[k, n] == weight[n, k] with stride_bk=stride(1), stride_bn=stride(0)
    # -------------------------------------------------------------------------
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_M, BLOCK_K)

    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )  # (BLOCK_K, BLOCK_N)

    # Accumulator (fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # K loop
    # -------------------------------------------------------------------------
    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        k_mask = offs_k < k_remaining

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused elementwise operations
    # All fused ops share the SAME offsets/masks (offs_m, offs_n, m_mask, n_mask)
    # -------------------------------------------------------------------------
    # Bias: [N] broadcast along M
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # y = min(acc, constant) - constant
    # Implemented as: t = acc - constant; y = min(t, 0)
    acc = acc - constant
    acc = tl.minimum(acc, 0.0)

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------
    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_min_sub(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    constant: torch.Tensor,
) -> torch.Tensor:
    """
    x:        [M, K]
    weight:   [N, K] (same layout as nn.Linear.weight)
    bias:     [N]
    constant: scalar tensor (0-d)
    Returns:  [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "Weight shape must be [N, K] where K == in_features"
    assert bias.shape[0] == N, "Bias shape must be [N]"

    # Output in fp32 to match accumulator precision
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    const_value = float(constant.item())

    # Grid: 1D over tiles of the [M, N] output
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_min_sub_kernel[grid](
        x,
        weight,                  # directly use [N, K] without transposing
        bias,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),        # stride_bk: along K (inner dim)
        weight.stride(0),        # stride_bn: along N (outer dim)
        c.stride(0),
        c.stride(1),
        const_value,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton implementation of:
        y = linear(x)
        y = torch.min(y, constant)
        y = y - constant
    """

    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.constant = nn.Parameter(torch.tensor(float(constant)))

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        return fused_linear_min_sub(x, self.weight, self.bias, self.constant)
