import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, D_in, H_in, W_in,
    C_out, K_D, K_H, K_W,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # flattened (N, D_out, H_out, W_out)
    BLOCK_N: tl.constexpr,  # output channels
    BLOCK_K: tl.constexpr,  # flattened (C_in, K_D, K_H, K_W)
):
    # ------------------------------
    # 2D grid over:
    #   M = N * D_out * H_out * W_out
    #   N = C_out
    # Inner loop over K = C_in * K_D * K_H * K_W (implicit GEMM)
    # ------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * D_out * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode flattened output index -> (n, od, oh, ow)
    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    od_idx = rem // HW_out
    rem2 = rem % HW_out
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Flattened K dimension
    K_HW = K_H * K_W
    K_DHW = K_D * K_HW
    K_total = C_in * K_DHW

    # GEMM-style loop over K with BLOCK_K
    for k0 in range(0, K_total, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_ids < K_total

        # Decode flattened k -> (c, kd, kh, kw)
        c_idx = k_ids // K_DHW
        remk = k_ids % K_DHW
        kd_idx = remk // K_HW
        remk2 = remk % K_HW
        kh_idx = remk2 // K_W
        kw_idx = remk2 % K_W

        # Compute input coordinates for this (M,K) tile
        id_idx = od_idx[:, None] + kd_idx[None, :]
        ih_idx = oh_idx[:, None] + kh_idx[None, :]
        iw_idx = ow_idx[:, None] + kw_idx[None, :]

        # Pointers for x: [M, K]
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + c_idx[None, :] * stride_xc
            + id_idx * stride_xd
            + ih_idx * stride_xh
            + iw_idx * stride_xw
        )
        # For our "valid" conv, spatial indices are always in-bounds when mask_m is true.
        # We still guard against overlaunch in M and K.
        mask_x = mask_m[:, None] & mask_k[None, :]

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        x = x.to(tl.float32)

        # Pointers for w: [K, N]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + c_idx[:, None] * stride_wc
            + kd_idx[:, None] * stride_wd
            + kh_idx[:, None] * stride_wh
            + kw_idx[:, None] * stride_ww
        )
        mask_w = mask_k[:, None] & mask_n[None, :]

        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        w = w.to(tl.float32)

        # Matrix multiply accumulate: (M, K) x (K, N) -> (M, N)
        acc += tl.dot(x, w)

    # Add bias (fused, same offsets/masks as weight loads / stores)
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    mask_yn = mask_m[:, None] & mask_n[None, :]
    # Cast back to original dtype (x/y/bias share dtype)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_yn)


@triton.jit
def groupnorm_min_clamp_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    N, C, D, H, W,
    groups, group_size,
    eps, min_value, max_value,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)
    ngroups_total = N * groups
    if pid >= ngroups_total:
        return

    n = pid // groups
    g = pid % groups

    # Total elements in this group
    DHW = D * H * W
    numel = group_size * DHW
    numel_f = tl.full((), numel, tl.float32)

    # ------------------------------
    # First pass: compute mean and variance
    # ------------------------------
    sum_val = tl.zeros((), tl.float32)
    sum_sq = tl.zeros((), tl.float32)

    for base in range(0, numel, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        # Decode flattened index within group -> (c_rel, d, h, w)
        c_rel = offs // DHW
        rem = offs % DHW
        d = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        c = g * group_size + c_rel

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / numel_f
    var = sum_sq / numel_f - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # ------------------------------
    # Second pass: normalize + affine + min + clamp
    # All fused ops share the SAME offsets & mask.
    # ------------------------------
    for base in range(0, numel, BLOCK_SIZE):
        offs = base + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        # Decode indices as above (MUST match first pass)
        c_rel = offs // DHW
        rem = offs % DHW
        d = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        c = g * group_size + c_rel

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Affine parameters per-channel
        gamma = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

        # GroupNorm
        x_hat = (x - mean) * rstd
        y = x_hat * gamma + beta

        # Fused min + clamp:
        # y = torch.min(y, min_value)
        # y = torch.clamp(y, min=min_value, max=max_value)
        # => effectively all outputs clamped to min_value, but we keep the exact sequence
        y = tl.minimum(y, min_value)
        y = tl.maximum(y, min_value)
        y = tl.minimum(y, max_value)

        y_ptrs = (
            y_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        tl.store(y_ptrs, y.to(tl.float32), mask=mask)


def conv3d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    3D convolution (N, C_in, D, H, W) with weight (C_out, C_in, K_D, K_H, K_W),
    stride=1, padding=0, dilation=1. Bias is (C_out,).
    Implemented as an implicit GEMM with tiling over (M, N, K).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in_w == C_in

    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Tuned for Ada (4090): 64x64 tiles, moderate K-block
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(N * D_out * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv3d_forward_kernel[grid](
        x, weight, bias, y,
        N, C_in, D_in, H_in, W_in,
        C_out, K_D, K_H, K_W,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return y


def groupnorm_min_clamp_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
    min_value: float,
    max_value: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    GroupNorm + min + clamp for NCDHW tensor x.
    GroupNorm(groups, C) with affine weight/bias.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C, D, H, W = x.shape
    assert weight.shape[0] == C and bias.shape[0] == C
    assert C % groups == 0
    group_size = C // groups

    y = torch.empty_like(x, dtype=torch.float32)

    BLOCK_SIZE = 256

    grid = lambda META: (N * groups,)

    groupnorm_min_clamp_kernel[grid](
        x, y,
        weight, bias,
        N, C, D, H, W,
        groups, group_size,
        float(eps), float(min_value), float(max_value),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-backed version of the reference model:
    Conv3d -> GroupNorm -> min -> clamp -> Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        # Use PyTorch modules only to own parameters; computation is done via Triton
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv3d via Triton
        y = conv3d_triton(x, self.conv.weight, self.conv.bias)
        # GroupNorm + min + clamp via Triton
        y = groupnorm_min_clamp_triton(
            y,
            self.norm.weight,
            self.norm.bias,
            self.norm.num_groups,
            self.min_value,
            self.max_value,
            self.norm.eps,
        )
        # Dropout via PyTorch (preserves training/eval behavior)
        y = self.dropout(y)
        return y
