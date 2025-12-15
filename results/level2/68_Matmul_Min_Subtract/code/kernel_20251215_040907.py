import math
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
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_min_sub_kernel(
    a_ptr,  # [M, K] input
    b_ptr,  # [K, N] weight^T
    bias_ptr,  # [N]
    c_ptr,  # [M, N] output
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    constant,  # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M x N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_M, BLOCK_K)
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )  # (BLOCK_K, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias: bias is [N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Fused elementwise: y = min(acc, constant) - constant
    # constant is scalar; broadcasting is applied automatically.
    acc = tl.minimum(acc, constant)
    acc = acc - constant

    # Store result
    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_min_sub(x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: torch.Tensor,
                         constant: torch.Tensor) -> torch.Tensor:
    """
    x:       [M, K]
    weight:  [N, K] (same as nn.Linear.weight)
    bias:    [N]
    constant: scalar tensor (0-d)
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"

    M, K = x.shape
    N = weight.shape[0]
    # We expect weight shape [N, K]
    assert weight.shape[1] == K, "Weight shape must be [N, K] where K == in_features"
    assert bias.shape[0] == N, "Bias shape must be [N]"

    # Make B = weight^T, shape [K, N]
    b = weight.t().contiguous()

    # Allocate output
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Extract scalar constant as Python float (assume float32)
    const_value = float(constant.item())

    # Grid
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_min_sub_kernel[grid](
        x,
        b,
        bias,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
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
