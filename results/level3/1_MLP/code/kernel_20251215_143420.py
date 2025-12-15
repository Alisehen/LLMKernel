import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced, conservative tile – good default
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4},
            num_warps=4,
            num_stages=2,
        ),
        # Same tile, more warps – better latency hiding when registers allow
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4},
            num_warps=8,
            num_stages=2,
        ),
        # Smaller M-tile – safer when register pressure is high
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_bias_kernel(
    a_ptr,        # [M, K]
    w_ptr,        # [N, K] (PyTorch Linear.weight, used as B[k, n] = w[n, k])
    b_ptr,        # [N]
    c_ptr,        # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,  # w[k, n] -> w_ptr + k*stride_wk + n*stride_wn
    stride_cm, stride_cn,
    HAS_RELU: tl.constexpr,   # whether to apply ReLU in epilogue
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,    # how many M-tiles to group for better L2 reuse
):
    # 1D launch grid with grouping along M to improve weight (B) reuse in L2
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group // num_pid_n
    pid_n = pid_in_group % num_pid_n

    # Some programs in the last group may be out-of-bounds in M
    if pid_m >= num_pid_m:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop, software-pipelined by Triton (num_stages set in autotune configs)
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k[None, :] < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask.T & (offs_n[None, :] < N),
            other=0.0,
        )
        # Use tensor cores via TF32 where possible
        acc += tl.dot(a, w, allow_tf32=True)

        # Advance pointers to next K-tile
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Epilogue: add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Optional ReLU, compiled out when HAS_RELU is False
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K] (PyTorch Linear.weight)
    bias:   [N]
    returns: [M, N] with ReLU
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    stride_am, stride_ak = x.stride()
    # Interpret weight as B[k, n] = weight[n, k]
    stride_wn = weight.stride(0)
    stride_wk = weight.stride(1)
    stride_cm, stride_cn = out.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

    _linear_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        HAS_RELU=True,
    )
    return out


def linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K] (PyTorch Linear.weight)
    bias:   [N]
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    stride_am, stride_ak = x.stride()
    stride_wn = weight.stride(0)
    stride_wk = weight.stride(1)
    stride_cm, stride_cn = out.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

    _linear_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        HAS_RELU=False,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()

        # Same structure as the original model for state_dict compatibility:
        # [Linear, ReLU, Linear, ReLU, ..., Linear]
        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fallback to original PyTorch behavior on CPU (Triton requires CUDA)
        if not x.is_cuda:
            return self.network(x)

        # Hidden layers: fuse Linear + ReLU using Triton
        num_layers = len(self.network)
        # Pattern: even indices (0, 2, ..., num_layers-3) are Linear with ReLU after them
        for idx in range(0, num_layers - 1, 2):
            linear = self.network[idx]
            x = fused_linear_relu_triton(x, linear.weight, linear.bias)

        # Final Linear layer (no ReLU)
        final_linear = self.network[-1]
        x = linear_triton(x, final_linear.weight, final_linear.bias)
        return x
