import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------
# Optimized GEMM (Linear) Kernel
# ------------------------


@triton.autotune(
    configs=[
        # Large tiles for high arithmetic intensity / tensor core utilization
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        # Fallbacks for small / skinny shapes and lower register pressure
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
    a_ptr,        # [M, K]
    w_ptr,        # [N, K] (nn.Linear.weight)
    bias_ptr,     # [N]
    c_ptr,        # [M, N]
    M, N, K,
    stride_am, stride_ak,  # A strides
    stride_wn, stride_wk,  # W strides (N, K)
    stride_cm, stride_cn,  # C strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for output tile [BLOCK_M, BLOCK_N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Pointers for tiles of A: [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Pointers for tiles of W: load as [BLOCK_N, BLOCK_K] (contiguous along K),
    # then transpose inside the dot to get [BLOCK_K, BLOCK_N].
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K

        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # w_tile: [BLOCK_N, BLOCK_K]
        w_tile = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Transpose W tile so that dot sees [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, tl.trans(w_tile), allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Add bias (broadcast along M dimension)
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Store final result (single store; all intermediate results stayed in registers)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=mask_out)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K] (nn.Linear.weight)
    bias:   [N]
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dim() == 2 and weight.dim() == 2
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w
    assert bias.numel() == N

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


# ------------------------
# Optimized GroupNorm + HardTanh Kernel
# ------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=1),
    ],
    key=["group_size"],
)
@triton.jit
def groupnorm_hardtanh_kernel(
    x_ptr,        # [B, C]
    weight_ptr,   # [C] (gamma)
    bias_ptr,     # [C] (beta)
    y_ptr,        # [B, C]
    B, C, G, group_size,
    stride_xb, stride_xc,
    stride_yb, stride_yc,
    eps,          # float
    min_val,      # float
    max_val,      # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // G
    g = pid % G
    if b >= B:
        return

    offs_c = tl.arange(0, BLOCK_SIZE)
    group_start = g * group_size

    # --------- First pass: compute mean & variance ----------
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    for c0 in range(0, group_size, BLOCK_SIZE):
        idx = c0 + offs_c
        mask = idx < group_size

        chan_idx = group_start + idx
        x_ptrs = x_ptr + b * stride_xb + chan_idx * stride_xc
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)

    group_elems = tl.full((), group_size, dtype=tl.float32)
    mean = sum_val / group_elems
    var = sum_sq_val / group_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # --------- Second pass: normalize + affine + HardTanh ----------
    for c0 in range(0, group_size, BLOCK_SIZE):
        idx = c0 + offs_c
        mask = idx < group_size

        chan_idx = group_start + idx

        x_ptrs = x_ptr + b * stride_xb + chan_idx * stride_xc
        y_ptrs = y_ptr + b * stride_yb + chan_idx * stride_yc

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + chan_idx, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0).to(tl.float32)

        y = ((x - mean) * inv_std) * gamma + beta
        y = tl.minimum(tl.maximum(y, min_val), max_val)

        # Single final store: all intermediate ops stay in registers
        tl.store(y_ptrs, y, mask=mask)


def triton_groupnorm_hardtanh(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    min_val: float,
    max_val: float,
) -> torch.Tensor:
    """
    x:      [B, C]
    weight: [C] (gamma)
    bias:   [C] (beta)
    """
    assert x.is_cuda
    assert x.dim() == 2
    B, C = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups
    assert weight.numel() == C
    assert bias.numel() == C

    y = torch.empty_like(x)

    grid = lambda META: (B * num_groups,)

    groupnorm_hardtanh_kernel[grid](
        x,
        weight,
        bias,
        y,
        B,
        C,
        num_groups,
        group_size,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        eps,
        min_val,
        max_val,
    )
    return y


# ------------------------
# PyTorch Module Wrapper
# ------------------------


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
      - Linear (GEMM)
      - GroupNorm
      - HardTanh
    """

    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = self.gemm(x)
            x = self.group_norm(x)
            x = self.hardtanh(x)
            return x

        x = triton_linear(x, self.gemm.weight, self.gemm.bias)
        x = triton_groupnorm_hardtanh(
            x,
            num_groups=self.group_norm.num_groups,
            weight=self.group_norm.weight,
            bias=self.group_norm.bias,
            eps=self.group_norm.eps,
            min_val=self.hardtanh.min_val,
            max_val=self.hardtanh.max_val,
        )
        return x
