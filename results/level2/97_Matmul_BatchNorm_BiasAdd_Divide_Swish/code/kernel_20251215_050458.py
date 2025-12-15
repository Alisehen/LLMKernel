import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Larger tiles, more warps – good for large GEMMs
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4},
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
    extra_bias_ptr,       # scalar [1]
    y_ptr,                # [M, N] output

    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    eps,
    divide_value,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program id & 2D tile mapping with M-grouping for better L2 locality
    # -------------------------------------------------------------------------
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_m = GROUP_M
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    pid_m = first_pid_m + (pid % group_m)
    pid_n = (pid % num_pid_in_group) // group_m

    # Offsets in output space
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base masks for M and N (shared by all fused ops)
    mask_m = offs_m < M
    mask_n = offs_n < N
    y_mask = mask_m[:, None] & mask_n[None, :]

    # -------------------------------------------------------------------------
    # Matmul: accum = x @ w    (x: [M, K], w: [K, N])
    # -------------------------------------------------------------------------
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        x_mask = mask_m[:, None] & k_mask[None, :]
        w_mask = k_mask[:, None] & mask_n[None, :]

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused epilogue: Linear bias + BatchNorm (inference) + extra bias + div
    # + Swish, all on the same [offs_m, offs_n] tile with shared mask.
    #
    # Algebraic fusion (per feature j):
    #   mat = (x @ W)_ij
    #   lin = mat + lin_bias_j
    #   bn  = gamma_j * (lin - mean_j) / sqrt(var_j + eps) + beta_j
    #   add = bn + extra_bias
    #   div = add / divide_value
    #   y   = div * sigmoid(div)
    #
    # We rewrite:
    #   scale_j = gamma_j / sqrt(var_j + eps)
    #   shift_j = beta_j - mean_j * scale_j
    #   scale_div_j = scale_j / divide_value
    #   bias_div_j  = (lin_bias_j * scale_j + shift_j + extra_bias) / divide_value
    #
    #   div = mat * scale_div_j + bias_div_j
    # -------------------------------------------------------------------------
    # Load per-feature (N-dim) parameters
    lin_bias = tl.load(lin_bias_ptr + offs_n, mask=mask_n, other=0.0)
    gamma = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=0.0)
    beta = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)
    running_mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    running_var = tl.load(running_var_ptr + offs_n, mask=mask_n, other=0.0)

    var_eps = running_var + eps
    inv_std = 1.0 / tl.sqrt(var_eps)
    scale = gamma * inv_std
    shift = beta - running_mean * scale

    extra_bias_val = tl.load(extra_bias_ptr)

    scale_div = scale / divide_value
    bias_div = (lin_bias * scale + shift + extra_bias_val) / divide_value

    acc = acc * scale_div[None, :] + bias_div[None, :]

    # -------------------------------------------------------------------------
    # Swish activation: x * sigmoid(x) = x / (1 + exp(-x))
    # -------------------------------------------------------------------------
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sig = 1.0 / (1.0 + exp_neg)
    acc = acc * sig

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
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
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
    assert x.dtype == torch.float32, "Kernel currently supports float32 only for x"
    assert weight.dtype == torch.float32, "Kernel currently supports float32 only for weight"

    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "in_features mismatch between input and weight"

    assert lin_bias.shape[0] == N
    assert bn_weight.shape[0] == N
    assert bn_bias.shape[0] == N
    assert running_mean.shape[0] == N
    assert running_var.shape[0] == N
    assert extra_bias.numel() == 1, "Kernel currently supports scalar extra_bias only"

    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    lin_bias_contig = lin_bias.contiguous()
    bn_weight_contig = bn_weight.contiguous()
    bn_bias_contig = bn_bias.contiguous()
    running_mean_contig = running_mean.contiguous()
    running_var_contig = running_var.contiguous()
    extra_bias_contig = extra_bias.contiguous()

    # Use weight^T inside kernel: [K, N]
    w_t = weight_contig.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
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
        self.bn_momentum = bn_momentum
        self.divide_value = float(divide_value)

        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.lin_bias = nn.Parameter(torch.zeros(out_features))

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.lin_bias, -bound, bound)

        # BatchNorm1d parameters (inference-style)
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))

        # Extra bias (broadcasted scalar)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
