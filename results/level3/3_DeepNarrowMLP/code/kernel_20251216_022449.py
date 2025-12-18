import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 4,
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 8,
            }
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_bias_relu_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]  (weight transposed)
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

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

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _run_linear_bias_relu(x, weight, bias, apply_relu: bool):
    """
    x: [M, K]
    weight: [N, K] (PyTorch Linear.weight, shape [out_features, in_features])
    bias: [N]
    """
    # Fallback to PyTorch on CPU or non-CUDA tensors
    if not x.is_cuda:
        y = F.linear(x, weight, bias)
        if apply_relu:
            y = F.relu(y)
        return y

    assert x.dim() == 2, "Input must be 2D (batch_size, features)"
    M, K = x.shape
    N = weight.shape[0]
    assert K == weight.shape[1], "Incompatible shapes for x and weight"

    # Triton kernel expects B as [K, N]
    b_t = weight.t().contiguous()

    # Allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = b_t.stride(0)
    stride_bn = b_t.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    # Grid: one program per BLOCK_M x BLOCK_N tile
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_relu_kernel[grid](
        x,
        b_t,
        bias,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        APPLY_RELU=apply_relu,
    )
    return c


def fused_linear_relu(x, weight, bias):
    return _run_linear_bias_relu(x, weight, bias, apply_relu=True)


def fused_linear(x, weight, bias):
    return _run_linear_bias_relu(x, weight, bias, apply_relu=False)


class TritonLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that uses a fused Triton kernel.
    Optionally applies ReLU inside the kernel.
    """

    def __init__(self, in_features, out_features, apply_relu: bool):
        super().__init__()
        # Use standard PyTorch Linear for parameter storage & initialization
        self.linear = nn.Linear(in_features, out_features)
        self.apply_relu = apply_relu

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        if self.apply_relu:
            return fused_linear_relu(x, self.weight, self.bias)
        else:
            return fused_linear(x, self.weight, self.bias)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()

        layers = []
        current_input_size = input_size

        # Hidden layers: Linear + ReLU fused
        for hidden_size in hidden_layer_sizes:
            layers.append(TritonLinear(current_input_size, hidden_size, apply_relu=True))
            current_input_size = hidden_size

        # Output layer: Linear only
        layers.append(TritonLinear(current_input_size, output_size, apply_relu=False))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)
