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
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=8,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    bias_ptr,  # (N,)
    c_ptr,  # (M, N)
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)  - same as nn.Linear(out_features=N, in_features=K).weight
    bias: (N,)
    returns: (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype
    M, K = x.shape
    N = weight.shape[0]

    # Triton kernel expects B as (K, N)
    b = weight.t().contiguous()
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
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
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Mirror of the original Model, but the final Linear is implemented via a Triton kernel.
        """
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        # Replace nn.Linear with explicit parameters to use Triton kernel
        self.weight = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x, h0, c0):
        """
        x: (batch_size, sequence_length, input_size)
        h0: (num_layers, batch_size, hidden_size)
        c0: (num_layers, batch_size, hidden_size)

        Returns:
            h_n (num_layers, batch_size, hidden_size) to match the original Model.
        """
        out, state = self.lstm(x, (h0, c0))  # out: (B, T, H)

        # Decode last time-step with Triton linear (computed but not returned,
        # to preserve original Model behavior).
        last = out[:, -1, :].contiguous()  # (B, H)
        _ = fused_linear(last, self.weight, self.bias)

        # Original Model returns state[0] (h_n)
        return state[0]
