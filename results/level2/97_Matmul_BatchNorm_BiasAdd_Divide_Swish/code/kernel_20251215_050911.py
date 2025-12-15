# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Larger tiles for big GEMMs
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Deeper K tile for more compute intensity
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=4,
        ),
        # Smaller tiles for edge cases / small shapes
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_bn_bias_div_swish_kernel(
    x_ptr,                # [M, K]
    w_ptr,                # [K, N]  (weight^T)
    lin_bias_ptr,         # [N]
    bn_weight_ptr,        # [N]
    bn_bias_ptr,          # [N]
    running_mean_ptr,     # [N]
    running_var_ptr,      # [N]
    extra_bias_ptr,       # scalar (numel == 1)
    y_ptr,                # [M, N] output

    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    eps,                  # batchnorm epsilon
    divide_value,         # scalar division value

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets within the overall matrices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the current tile of X and W
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matrix multiply accumulate
        acc += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # ---- Fused epilogue: Linear bias + BatchNorm + extra bias + division + Swish ----
    # All operations stay in registers; only one final tl.store at the end.

    n_mask = offs_n < N

    # Load per-feature parameters
    lin_bias = tl.load(lin_bias_ptr + offs_n, mask=n_mask, other=0.0)
    gamma = tl.load(bn_weight_ptr + offs_n, mask=n_mask, other=0.0)
    beta = tl.load(bn_bias_ptr + offs_n, mask=n_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + offs_n, mask=n_mask, other=0.0)
    running_var = tl.load(running_var_ptr + offs_n, mask=n_mask, other=1.0)

    # BatchNorm scale and shift (inference-style)
    # inv_std = 1/sqrt(var + eps)
    var_eps = running_var + eps
    inv_std = 1.0 / tl.sqrt(var_eps)
    scale = gamma * inv_std            # per-feature scale

    # Combine linear bias into BN shift to avoid an extra add in the hot loop:
    # Original:
    #   acc0 = xW
    #   acc1 = acc0 + lin_bias
    #   acc2 = scale * acc1 + (beta - mean * scale)
    #
    # Combined:
    #   acc2 = scale * acc0 + (beta - mean * scale + scale * lin_bias)
    combined_shift = beta - running_mean * scale + lin_bias * scale

    # Extra scalar bias and division folded into final affine transform:
    #   y = ((acc2 + extra_bias) / divide_value)
    #     = (scale / divide_value) * acc0 +
    #       (combined_shift + extra_bias) / divide_value
    extra_bias_val = tl.load(extra_bias_ptr)
    inv_div = 1.0 / divide_value

    final_scale = scale * inv_div
    final_shift = (combined_shift + extra_bias_val) * inv_div

    # Apply final affine transform
    acc = acc * final_scale[None, :] + final_shift[None, :]

    # Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sig = 1.0 / (1.0 + exp_neg)
    acc = acc * sig

    # Store final result (single store, no intermediate writes)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


def fused_linear_bn_bias_div_swish(
    x: torch.Tensor,
    weight: torch.Tensor,
    lin_bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    extra_bias: torch.Tensor,
    eps: float,
    divide_value: float,
) -> torch.Tensor:
    """
    Fused implementation of:
        y = x @ weight.T + lin_bias
        y = BatchNorm1d(y; running_mean, running_var, bn_weight, bn_bias, eps)  (inference-style)
        y = y + extra_bias          # scalar bias
        y = y / divide_value
        y = y * sigmoid(y)          # Swish

    Shapes:
        x:          [M, K]
        weight:     [N, K]
        lin_bias:   [N]
        bn_weight:  [N]
        bn_bias:    [N]
        running_*:  [N]
        extra_bias: [1] (scalar)
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda, "Weight must be on CUDA device"
    assert x.dtype == torch.float32, "Kernel currently supports float32 only"
    assert weight.dtype == torch.float32, "Kernel currently supports float32 only"

    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "in_features mismatch between input and weight"

    assert lin_bias.shape[0] == N
    assert bn_weight.shape[0] == N
    assert bn_bias.shape[0] == N
    assert running_mean.shape[0] == N
    assert running_var.shape[0] == N
    assert extra_bias.numel() == 1, "Kernel currently supports scalar extra_bias only"

    # Make sure tensors are contiguous where it matters
    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    lin_bias_contig = lin_bias.contiguous()
    bn_weight_contig = bn_weight.contiguous()
    bn_bias_contig = bn_bias.contiguous()
    running_mean_contig = running_mean.contiguous()
    running_var_contig = running_var.contiguous()
    extra_bias_contig = extra_bias.contiguous()

    # We use weight^T inside the kernel: [K, N]
    w_t = weight_contig.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    fused_linear_bn_bias_div_swish_kernel[grid](
        x_contig,
        w_t,
        lin_bias_contig,
        bn_weight_contig,
        bn_bias_contig,
        running_mean_contig,
        running_var_contig,
        extra_bias_contig,
        y,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        float(eps),
        float(divide_value),
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs:
        Linear -> BatchNorm1d (inference-style) -> bias add -> division -> Swish
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,  # unused, kept for API compatibility
        bias_shape=(1,),
        divide_value: float = 1.0,
    ):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum  # not used (no running-stat updates in this fused version)
        self.divide_value = float(divide_value)

        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.lin_bias = nn.Parameter(torch.zeros(out_features))

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.lin_bias, -bound, bound)

        # BatchNorm1d parameters (inference-style)
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        # Extra bias (broadcasted scalar)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, in_features]
        assert x.dim() == 2, "Expected input of shape [batch_size, in_features]"
        assert x.shape[1] == self.in_features, "in_features mismatch"

        return fused_linear_bn_bias_div_swish(
            x,
            self.weight,
            self.lin_bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.bias,
            self.bn_eps,
            self.divide_value,
        )
