import math
import torch, torch.nn as nn, triton, triton.language as tl


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
    # Program ids for 2D tiling of output matrix C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col indices for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 for better numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_remaining = K
    while k_remaining > 0:
        # Mask K for this tile
        k_mask = offs_k[None, :] < k_remaining

        # Load A and B tiles (masked for M, N, and K)
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

        # Fused matmul tile
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    # Add bias: shape (N,)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if ADD_RELU:
        acc = tl.maximum(acc, 0.0)

    # Write back, cast to original dtype of C
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

    # Ensure contiguous for expected strides
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
        b.stride(1), b.stride(0),  # b is (N, K) so K-major for kernel: stride_bk, stride_bn
        c.stride(0), c.stride(1),
        ADD_RELU=add_relu,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return c


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
        # Move input to same device/dtype as parameters if needed
        if self.weights:
            param = self.weights[0]
            if x.device != param.device:
                x = x.to(param.device)
            if x.dtype != param.dtype:
                x = x.to(param.dtype)

        x = x.contiguous()

        num_layers = len(self.weights)
        for i in range(num_layers):
            w = self.weights[i]          # (out_features, in_features)
            b = self.biases[i]           # (out_features,)
            # Kernel expects weight as (N, K) with K = in_features
            w_t = w                      # shape (out_features, in_features)
            # Use Triton kernel: all but last layer have ReLU
            add_relu = i < (num_layers - 1)
            x = triton_linear(x, w_t, b, add_relu=add_relu)
        return x
