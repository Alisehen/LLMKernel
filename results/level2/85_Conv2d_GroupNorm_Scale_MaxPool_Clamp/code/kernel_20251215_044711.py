import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # over P = N * H_out * W_out
    BLOCK_N: tl.constexpr,  # over C_out
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    P = N * H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Naive direct convolution (stride=1, padding=0, dilation=1)
    for ic in range(0, C_in):
        for kh in range(0, K_H):
            h_in = oh_idx + kh
            for kw in range(0, K_W):
                w_in = ow_idx + kw

                x_ptrs = x_ptr + (
                    n_idx * stride_xn
                    + ic * stride_xc
                    + h_in * stride_xh
                    + w_in * stride_xw
                )

                x_mask = (
                    mask_m
                    & (n_idx < N)
                    & (h_in < H_in)
                    & (w_in < W_in)
                    & (ic < C_in)
                )
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

                w_ptrs = w_ptr + (
                    offs_n * stride_wo
                    + ic * stride_wi
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)

                acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += b_vals[None, :]

    y_ptrs = y_ptr + (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_triton(x, weight, bias):
    """
    Simple NCHW conv2d with stride=1, padding=0, dilation=1 implemented in Triton.
    Args:
        x: (N, C_in, H_in, W_in)
        weight: (C_out, C_in, K_H, K_W)
        bias: (C_out,)
    Returns:
        y: (N, C_out, H_out, W_out)
    """
    assert x.ndim == 4 and weight.ndim == 4
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight"

    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert H_out > 0 and W_out > 0, "Invalid conv output size"

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    if bias is None:
        b_contig = torch.zeros(C_out, device=x.device, dtype=x.dtype)
    else:
        b_contig = bias.contiguous()

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(N * H_out * W_out, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_nchw_kernel[grid](
        x_contig, w_contig, b_contig, y,
        N, C_in, H_in, W_in,
        C_out, K_H, K_W,
        H_out, W_out,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        w_contig.stride(0), w_contig.stride(1), w_contig.stride(2), w_contig.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64, BLOCK_N=32,
    )
    return y


@triton.jit
def groupnorm_scale_kernel(
    x_ptr, gamma_ptr, beta_ptr, scale_ptr, y_ptr,
    N, C, H, W,
    num_groups, group_size,
    HW, group_elems,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # over N * num_groups
    n = pid // num_groups
    g = pid % num_groups

    valid_group = n < N
    c0 = g * group_size

    mean = tl.zeros((), dtype=tl.float32)
    m2 = tl.zeros((), dtype=tl.float32)

    # First pass: compute mean and variance over this (n, group)
    for offset in range(0, group_elems, BLOCK):
        offs = offset + tl.arange(0, BLOCK)
        mask = (offs < group_elems) & valid_group

        c_off = offs // HW
        rem = offs % HW
        h = rem // W
        w = rem % W
        c = c0 + c_off

        x_ptrs = x_ptr + (
            n * stride_xn
            + c * stride_xc
            + h * stride_xh
            + w * stride_xw
        )
        vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        block_sum = tl.sum(vals, axis=0)
        block_sumsq = tl.sum(vals * vals, axis=0)
        mean += block_sum
        m2 += block_sumsq

    L_f = tl.full((), group_elems, dtype=tl.float32)
    mean = mean / L_f
    var = m2 / L_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine (GroupNorm) + extra scale
    for offset in range(0, group_elems, BLOCK):
        offs = offset + tl.arange(0, BLOCK)
        mask = (offs < group_elems) & valid_group

        c_off = offs // HW
        rem = offs % HW
        h = rem // W
        w = rem % W
        c = c0 + c_off

        x_ptrs = x_ptr + (
            n * stride_xn
            + c * stride_xc
            + h * stride_xh
            + w * stride_xw
        )
        vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + c, mask=mask, other=1.0).to(tl.float32)

        norm = (vals - mean) * inv_std
        y_vals = norm * gamma + beta
        y_vals = y_vals * scale

        y_ptrs = y_ptr + (
            n * stride_yn
            + c * stride_yc
            + h * stride_yh
            + w * stride_yw
        )
        tl.store(y_ptrs, y_vals, mask=mask)


def groupnorm_scale_triton(x, num_groups, weight, bias, scale, eps=1e-5):
    """
    GroupNorm followed by channel-wise scale in Triton.

    Args:
        x: (N, C, H, W)
        num_groups: int
        weight: (C,)  GroupNorm gamma
        bias: (C,)    GroupNorm beta
        scale: (C, 1, 1) or broadcastable to (C, H, W)
    """
    assert x.ndim == 4
    N, C, H, W = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups
    HW = H * W
    group_elems = group_size * HW

    x_contig = x.contiguous()
    gamma = weight.contiguous().view(C)
    beta = bias.contiguous().view(C)
    scale_flat = scale.contiguous().view(C)

    y = torch.empty_like(x_contig)

    def grid(meta):
        return (N * num_groups,)

    groupnorm_scale_kernel[grid](
        x_contig, gamma, beta, scale_flat, y,
        N, C, H, W,
        num_groups, group_size,
        HW, group_elems,
        eps,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=128,
    )
    return y


@triton.jit
def maxpool_clamp_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    H_out, W_out,
    K, stride_hw,
    clamp_min, clamp_max,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # over P = N * H_out * W_out
    BLOCK_N: tl.constexpr,  # over C
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    P = N * H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    acc = tl.full((BLOCK_M, BLOCK_N), -float("inf"), dtype=tl.float32)

    for kh in range(0, K):
        h_in = oh_idx * stride_hw + kh
        for kw in range(0, K):
            w_in = ow_idx * stride_hw + kw

            x_ptrs = x_ptr + (
                n_idx[:, None] * stride_xn
                + offs_n[None, :] * stride_xc
                + h_in[:, None] * stride_xh
                + w_in[:, None] * stride_xw
            )

            mask_hw = (h_in[:, None] < H) & (w_in[:, None] < W)
            mask = mask_m[:, None] & mask_n[None, :] & mask_hw

            vals = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            acc = tl.maximum(acc, vals)

    acc = tl.maximum(acc, clamp_min)
    acc = tl.minimum(acc, clamp_max)

    y_ptrs = y_ptr + (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def maxpool_clamp_triton(x, kernel_size, clamp_min, clamp_max, stride=None):
    """
    MaxPool2d with kernel_size and stride (default stride=kernel_size) followed by clamp.
    Args:
        x: (N, C, H, W)
    """
    if stride is None:
        stride = kernel_size

    assert x.ndim == 4
    N, C, H, W = x.shape

    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    assert H_out > 0 and W_out > 0

    x_contig = x.contiguous()
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(N * H_out * W_out, meta["BLOCK_M"]),
            triton.cdiv(C, meta["BLOCK_N"]),
        )

    maxpool_clamp_kernel[grid](
        x_contig, y,
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride,
        clamp_min, clamp_max,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64, BLOCK_N=32,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-based implementation of:
      Conv2d -> GroupNorm -> per-channel scale -> MaxPool2d -> clamp
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 num_groups, scale_shape, maxpool_kernel_size,
                 clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Conv parameters
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Match PyTorch Conv2d default initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) if in_channels > 0 else None
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # GroupNorm parameters
        self.num_groups = num_groups
        self.group_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_channels))

        # Extra per-channel scale
        self.scale = nn.Parameter(torch.ones(scale_shape))

        # MaxPool and clamp params
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Conv2d
        x = conv2d_triton(x, self.weight, self.bias)
        # GroupNorm + extra scale
        x = groupnorm_scale_triton(
            x,
            self.num_groups,
            self.group_norm_weight,
            self.group_norm_bias,
            self.scale,
            eps=1e-5,
        )
        # MaxPool + clamp
        x = maxpool_clamp_triton(
            x,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
            stride=self.maxpool_kernel_size,
        )
        return x


# Needed for initialization (kaiming_uniform_)
import math
