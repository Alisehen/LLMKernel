# <complete ModelNew code with optimized Triton kernels>
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
    ADD_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_remaining = K
    while k_remaining > 0:
        k_mask = offs_k[None, :] < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    if ADD_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out = acc.to(tl.float32)
    tl.store(
        c_ptrs,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, add_relu: bool):
    """
    x:      (M, K)
    weight: (N, K)  (nn.Linear's weight.t() contiguous)
    bias:   (N,)
    returns: (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape[1] == weight.shape[1], "Incompatible shapes"
    assert weight.shape[0] == bias.shape[0], "Weight/bias size mismatch"

    M, K = x.shape
    N = weight.shape[0]

    a = x.contiguous()
    b = weight.contiguous()
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        max(1, triton.cdiv(M, META["BLOCK_M"])),
        max(1, triton.cdiv(N, META["BLOCK_N"])),
    )

    linear_bias_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
        ADD_RELU=add_relu,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return c


@triton.jit
def fused_mlp3_kernel(
    x_ptr,          # (M, K0)
    w0_ptr, b0_ptr,
    w1_ptr, b1_ptr,
    w2_ptr, b2_ptr,
    buf0_ptr,       # (M, N0)
    buf1_ptr,       # (M, N1)
    out_ptr,        # (M, N2)
    M, K0, K1, K2,
    N0, N1, N2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused 3-layer MLP:
      Layer 0: x      @ w0^T + b0 -> ReLU -> buf0   (M x K0) -> (M x N0)
      Layer 1: buf0   @ w1^T + b1 -> ReLU -> buf1   (M x K1) -> (M x N1)
      Layer 2: buf1   @ w2^T + b2        -> out     (M x K2) -> (M x N2)

    Shapes (row-major, contiguous):
      w0: (N0, K0), b0: (N0,)
      w1: (N1, K1), b1: (N1,)
      w2: (N2, K2), b2: (N2,)
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # -----------------------------------------
    # Layer 0
    # -----------------------------------------
    N = N0
    K = K0
    mask_n = offs_n < N

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A: x_ptr with shape (M, K0), row-major -> stride = K0
    a_ptrs = x_ptr + offs_m[:, None] * K0 + offs_k[None, :]
    # B: w0_ptr with shape (N0, K0), row-major; we want B[k, n] = W0[n, k]
    b_ptrs = w0_ptr + offs_n[None, :] * K0 + offs_k[:, None]

    k_remaining = K
    while k_remaining > 0:
        k_mask = offs_k[None, :] < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
            other=0.0,
        )

        acc0 += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        k_remaining -= BLOCK_K

    bias0 = tl.load(b0_ptr + offs_n, mask=mask_n, other=0.0)
    acc0 += bias0[None, :]
    acc0 = tl.maximum(acc0, 0.0)  # ReLU

    c_ptrs0 = buf0_ptr + offs_m[:, None] * N0 + offs_n[None, :]
    tl.store(
        c_ptrs0,
        acc0.to(tl.float32),
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )

    # -----------------------------------------
    # Layer 1
    # -----------------------------------------
    N = N1
    K = K1
    mask_n = offs_n < N

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A: buf0 (M, K1), row-major -> stride = K1 (K1 == N0)
    a_ptrs = buf0_ptr + offs_m[:, None] * K1 + offs_k[None, :]
    # B: w1_ptr (N1, K1), row-major
    b_ptrs = w1_ptr + offs_n[None, :] * K1 + offs_k[:, None]

    k_remaining = K
    while k_remaining > 0:
        k_mask = offs_k[None, :] < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
            other=0.0,
        )

        acc1 += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        k_remaining -= BLOCK_K

    bias1 = tl.load(b1_ptr + offs_n, mask=mask_n, other=0.0)
    acc1 += bias1[None, :]
    acc1 = tl.maximum(acc1, 0.0)  # ReLU

    c_ptrs1 = buf1_ptr + offs_m[:, None] * N1 + offs_n[None, :]
    tl.store(
        c_ptrs1,
        acc1.to(tl.float32),
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )

    # -----------------------------------------
    # Layer 2 (no ReLU)
    # -----------------------------------------
    N = N2
    K = K2
    mask_n = offs_n < N

    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # A: buf1 (M, K2), row-major -> stride = K2 (K2 == N1)
    a_ptrs = buf1_ptr + offs_m[:, None] * K2 + offs_k[None, :]
    # B: w2_ptr (N2, K2), row-major
    b_ptrs = w2_ptr + offs_n[None, :] * K2 + offs_k[:, None]

    k_remaining = K
    while k_remaining > 0:
        k_mask = offs_k[None, :] < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_remaining),
            other=0.0,
        )

        acc2 += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
        k_remaining -= BLOCK_K

    bias2 = tl.load(b2_ptr + offs_n, mask=mask_n, other=0.0)
    acc2 += bias2[None, :]

    c_ptrs2 = out_ptr + offs_m[:, None] * N2 + offs_n[None, :]
    tl.store(
        c_ptrs2,
        acc2.to(tl.float32),
        mask=(offs_m[:, None] < M) & mask_n[None, :],
    )


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()

        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = nn.Parameter(torch.empty(out_features, in_features))
            b = nn.Parameter(torch.empty(out_features))
            self.weights.append(w)
            self.biases.append(b)

        self.reset_parameters()

    def reset_parameters(self):
        # Mimic nn.Linear initialization as closely as possible
        for w, b in zip(self.weights, self.biases):
            in_features = w.shape[1]
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if b is not None:
                fan_in = in_features
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        if self.weights:
            param = self.weights[0]
            if x.device != param.device:
                x = x.to(param.device)
            if x.dtype != param.dtype:
                x = x.to(param.dtype)

        x = x.contiguous()
        num_layers = len(self.weights)

        # Optimized fused path for exactly 3 linear layers
        if num_layers == 3 and x.is_cuda:
            w0, w1, w2 = [w.contiguous() for w in self.weights]
            b0, b1, b2 = self.biases

            M = x.shape[0]
            K0 = x.shape[1]
            N0 = w0.shape[0]

            K1 = w1.shape[1]
            N1 = w1.shape[0]

            K2 = w2.shape[1]
            N2 = w2.shape[0]

            # Enforce connectivity: in_features of each layer = out_features of previous
            assert K0 == w0.shape[1], "Layer 0 weight shape mismatch"
            assert K1 == N0, "Layer 1 in_features must match Layer 0 out_features"
            assert K2 == N1, "Layer 2 in_features must match Layer 1 out_features"

            buf0 = torch.empty((M, N0), device=x.device, dtype=x.dtype)
            buf1 = torch.empty((M, N1), device=x.device, dtype=x.dtype)
            out = torch.empty((M, N2), device=x.device, dtype=x.dtype)

            max_N = max(N0, N1, N2)

            grid = lambda META: (
                max(1, triton.cdiv(M, META["BLOCK_M"])),
                max(1, triton.cdiv(max_N, META["BLOCK_N"])),
            )

            fused_mlp3_kernel[grid](
                x,
                w0, b0,
                w1, b1,
                w2, b2,
                buf0, buf1,
                out,
                M, K0, K1, K2,
                N0, N1, N2,
                BLOCK_M=64,
                BLOCK_N=128,
                BLOCK_K=32,
                num_warps=4,
                num_stages=2,
            )
            return out

        # Generic fallback: sequential Triton linear+bias+ReLU kernels
        out = x
        for i in range(num_layers):
            w = self.weights[i].contiguous()  # (out_features, in_features)
            b = self.biases[i]
            add_relu = i < (num_layers - 1)
            out = triton_linear(out, w, b, add_relu=add_relu)
        return out
