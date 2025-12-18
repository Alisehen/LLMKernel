import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_bias_relu_kernel(
    a_ptr,       # [M, K]  (row-major)
    b_ptr,       # [N, K]  (PyTorch Linear.weight, row-major)
    bias_ptr,    # [N]
    c_ptr,       # [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,   # stride of B along N dimension (weight.stride(0))
    stride_bk,   # stride of B along K dimension (weight.stride(1))
    stride_cm,
    stride_cn,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program id: grid covers output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets into M and N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.max_contiguous(offs_m, BLOCK_M)
    tl.max_contiguous(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Base masks for M and N dimensions (shared by all fused ops)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # Pointers for the first K-tile
    # A: [M, K], row-major -> stride_am (between rows), stride_ak (between cols)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B is [N, K] but we access it as B^T with shape [K, N] for tl.dot:
    # B_T[k, n] = B[n, k]
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    # Accumulator in fp32 for numerical stability and Tensor Core usage
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        # Masks along K dimension for current tile
        k_mask_row = offs_k[None, :] < k_remaining          # (1, BLOCK_K) for A
        k_mask_col = offs_k[:, None] < k_remaining          # (BLOCK_K, 1) for B

        a_mask = mask_m & k_mask_row                        # (BLOCK_M, BLOCK_K)
        b_mask = k_mask_col & mask_n                        # (BLOCK_K, BLOCK_N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Use Tensor Cores where possible (TF32 for fp32, HMMA for fp16/bf16)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        # Move to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Bias add: uses same offsets/mask in N as matmul output
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU on the same output tile
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store result with a single, shared boundary mask
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m & mask_n
    tl.store(c_ptrs, acc, mask=c_mask)


def _run_linear_bias_relu(x, weight, bias, apply_relu: bool):
    """
    x:      [M, K]
    weight: [N, K] (nn.Linear.weight)
    bias:   [N]
    """
    if not x.is_cuda:
        out = torch.nn.functional.linear(x, weight, bias)
        if apply_relu:
            out = torch.nn.functional.relu(out)
        return out

    # Ensure contiguous for optimal memory access
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    assert K == weight.shape[1], "Incompatible shapes for x and weight"

    # Output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    # weight: [N, K] row-major
    stride_bn = weight.stride(0)  # along N (rows)
    stride_bk = weight.stride(1)  # along K (cols)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_relu_kernel[grid](
        x,
        weight,
        bias,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
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
