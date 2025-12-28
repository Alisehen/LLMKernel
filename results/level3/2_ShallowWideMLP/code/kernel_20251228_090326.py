# <complete ModelNew code with optimized Triton kernels>
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program ids for [M, N] tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for M, N, K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundaries
    m_mask = offs_m < M
    n_mask = offs_n < N
    out_mask = m_mask[:, None] & n_mask[None, :]

    # Compiler hints
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Base pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ----- FUSED EPILOGUE (bias + optional ReLU) -----
    # Load bias as [BLOCK_N] and broadcast across M
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)  # [BN]
    acc += bias[None, :]  # broadcast over M

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=out_mask)


def _linear_bias_grid(meta):
    # meta contains runtime sizes (M, N, K) and compile-time BLOCK_* from autotune
    return (
        triton.cdiv(meta['M'], meta['BLOCK_M']),
        triton.cdiv(meta['N'], meta['BLOCK_N']),
    )


def fused_linear_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + Bias + ReLU using Triton.
    x:      [M, K]
    weight: [K, N]  (stored as [in_features, out_features])
    bias:   [N]
    returns: [M, N] with bias + ReLU applied
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be on CUDA device"

    # Work in FP32 for numerical stability
    if x.dtype != torch.float32:
        x = x.float()
    if weight.dtype != torch.float32:
        weight = weight.float()
    if bias.dtype != torch.float32:
        bias = bias.float()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    K_w, N = weight.shape
    assert K_w == K, f"Incompatible shapes: x [{M}, {K}], weight [{K_w}, {N}]"

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    linear_bias_kernel[_linear_bias_grid](  # grid is 2D over [M, N]
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        APPLY_RELU=True,
    )
    return out


def fused_linear_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + Bias using Triton.
    x:      [M, K]
    weight: [K, N]  (stored as [in_features, out_features])
    bias:   [N]
    returns: [M, N] with bias applied
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be on CUDA device"

    if x.dtype != torch.float32:
        x = x.float()
    if weight.dtype != torch.float32:
        weight = weight.float()
    if bias.dtype != torch.float32:
        bias = bias.float()

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    K_w, N = weight.shape
    assert K_w == K, f"Incompatible shapes: x [{M}, {K}], weight [{K_w}, {N}]"

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    linear_bias_kernel[_linear_bias_grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        APPLY_RELU=False,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        Triton-accelerated MLP.
        Uses fused Linear+Bias+ReLU kernels for hidden layers
        and fused Linear+Bias for the output layer.

        We store weights in [in_features, out_features] layout
        to match Triton matmul directly (no transpose in forward).
        """
        super(ModelNew, self).__init__()

        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.num_layers = len(layer_sizes) - 1
        self.num_hidden = len(hidden_layer_sizes)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        # Initialize like nn.Linear (Kaiming uniform), but store [in, out]
        for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:]):
            # temp tensor in [out, in] for nn.init
            w_oi = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(w_oi, a=math.sqrt(5))
            # store as [in, out] for Triton
            w = nn.Parameter(w_oi.t().contiguous())

            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            b = nn.Parameter(torch.empty(out_features))
            nn.init.uniform_(b, -bound, bound)

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_size]
        returns: [batch_size, output_size]
        """
        assert x.is_cuda, "Input must be on CUDA device"
        if x.dtype != torch.float32:
            x = x.float()

        for i in range(self.num_layers):
            w = self.weights[i]   # [in_features, out_features]
            b = self.biases[i]    # [out_features]
            if i < self.num_hidden:
                x = fused_linear_bias_relu(x, w, b)
            else:
                x = fused_linear_bias(x, w, b)
        return x
