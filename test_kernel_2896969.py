import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling of the output matrix C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Pointers to the first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )

        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (1D over N)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if RELU:
        acc = tl.maximum(acc, 0.0)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, relu: bool):
    """
    x:      [M, K]
    weight: [N, K]  (same as nn.Linear.weight)
    bias:   [N]
    returns: [M, N]
    Computes y = x @ weight.T + bias (+ ReLU if relu=True)
    """
    assert x.ndim == 2 and weight.ndim == 2
    assert bias.ndim == 1
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert bias.shape[0] == N

    # Output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # We logically treat B as [K, N] with strides (stride_bk, stride_bn)
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = weight.stride(1)  # along K
    stride_bn = weight.stride(0)  # along N
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_kernel[grid](
        x, weight, bias, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        relu,  # RELU flag
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )

    return c


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        Triton-optimized MLP:
        [Linear + ReLU] x len(layer_sizes)  ->  Linear
        All Linear layers use a fused Triton GEMM(+bias[+ReLU]) kernel.
        """
        super(ModelNew, self).__init__()

        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()

        in_features = input_size
        for hidden_size in layer_sizes:
            w = nn.Parameter(torch.empty(hidden_size, in_features))
            b = nn.Parameter(torch.empty(hidden_size))
            # Initialize like nn.Linear
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

            self.hidden_weights.append(w)
            self.hidden_biases.append(b)
            in_features = hidden_size

        # Output layer
        self.out_weight = nn.Parameter(torch.empty(output_size, in_features))
        self.out_bias = nn.Parameter(torch.empty(output_size))
        nn.init.kaiming_uniform_(self.out_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.out_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hidden layers: Linear + ReLU
        for w, b in zip(self.hidden_weights, self.hidden_biases):
            x = fused_linear_bias(x, w, b, relu=True)

        # Output layer: Linear only
        x = fused_linear_bias(x, self.out_weight, self.out_bias, relu=False)
        return x
