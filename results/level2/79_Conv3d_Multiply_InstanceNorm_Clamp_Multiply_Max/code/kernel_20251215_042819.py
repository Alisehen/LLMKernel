# Triton implementation of Conv3d + multiply + InstanceNorm3d + clamp + multiply + channel max

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_ncdhw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in,
    D_in, H_in, W_in,
    C_out,
    K_D, K_H, K_W,
    D_out, H_out, W_out,
    DHW_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # positions (flattened N*D_out*H_out*W_out)
    BLOCK_N: tl.constexpr,  # output channels
):
    pid_m = tl.program_id(0)  # over positions
    pid_n = tl.program_id(1)  # over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * DHW_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    HW_out = H_out * W_out

    # Decode flattened position -> (n, od, oh, ow)
    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    od_idx = rem // HW_out
    rem2 = rem % HW_out
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution: loops over input channels and kernel volume
    for ic in range(0, C_in):
        for kd in range(0, K_D):
            for kh in range(0, K_H):
                for kw in range(0, K_W):
                    id_idx = od_idx + kd
                    ih_idx = oh_idx + kh
                    iw_idx = ow_idx + kw

                    # x has shape [N, C_in, D_in, H_in, W_in] in NCDHW layout
                    x_offsets = (
                        n_idx * stride_xn
                        + ic * stride_xc
                        + id_idx * stride_xd
                        + ih_idx * stride_xh
                        + iw_idx * stride_xw
                    )
                    x_ptrs = x_ptr + x_offsets
                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M]

                    # w has shape [C_out, C_in, K_D, K_H, K_W]
                    w_offsets = (
                        offs_n * stride_wn
                        + ic * stride_wc
                        + kd * stride_wd
                        + kh * stride_wh
                        + kw * stride_ww
                    )
                    w_ptrs = w_ptr + w_offsets
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]

                    acc += x_vals[:, None] * w_vals[None, :]

    # Add bias per output channel
    if b_ptr != 0:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    # Store result in y [N, C_out, D_out, H_out, W_out]
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def instancenorm_clamp_mul_kernel(
    x_ptr, multiplier_ptr, y_ptr,
    N, C, D, H, W,
    S, HW,                # S = D*H*W, HW = H*W
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_mc, stride_md, stride_mh, stride_mw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    eps, clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # over (n, c) pairs
    nc = pid
    n = nc // C
    c = nc % C

    offs = tl.arange(0, BLOCK_S)

    # Per-channel multiplier scalar (multiplier shape [C, 1, 1, 1])
    m_ptr = multiplier_ptr + c * stride_mc
    m_val = tl.load(m_ptr)

    # First pass: compute mean and variance over spatial positions for this (n, c)
    mean = tl.zeros((), dtype=tl.float32)
    mean_sq = tl.zeros((), dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_idx = s_start + offs
        mask = s_idx < S

        d = s_idx // HW
        rem = s_idx % HW
        h = rem // W
        w = rem % W

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_scaled = x * m_val

        mean += tl.sum(x_scaled, axis=0)
        mean_sq += tl.sum(x_scaled * x_scaled, axis=0)

    S_f = tl.float32(S)
    mean = mean / S_f
    mean_sq = mean_sq / S_f
    var = mean_sq - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, clamp, and multiply again
    for s_start in range(0, S, BLOCK_S):
        s_idx = s_start + offs
        mask = s_idx < S

        d = s_idx // HW
        rem = s_idx % HW
        h = rem // W
        w = rem % W

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_scaled = x * m_val
        x_norm = (x_scaled - mean) * inv_std
        x_clamped = tl.maximum(tl.minimum(x_norm, clamp_max), clamp_min)
        y_vals = x_clamped * m_val

        y_ptrs = (
            y_ptr
            + n * stride_yn
            + c * stride_yc
            + d * stride_yd
            + h * stride_yh
            + w * stride_yw
        )
        tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def channel_max_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    P, DHW, HW,          # P = N*D*H*W, DHW = D*H*W, HW = H*W
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_yn, stride_yd, stride_yh, stride_yw,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = offs < P

    n = offs // DHW
    rem = offs % DHW
    d = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    max_vals = tl.full((BLOCK_P,), -float("inf"), dtype=tl.float32)

    for c in range(0, C):
        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + d * stride_xd
            + h * stride_xh
            + w * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        max_vals = tl.maximum(max_vals, x_vals)

    y_ptrs = (
        y_ptr
        + n * stride_yn
        + d * stride_yd
        + h * stride_yh
        + w * stride_yw
    )
    tl.store(y_ptrs, max_vals, mask=mask)


def conv3d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Direct 3D convolution in NCDHW with kernel [C_out, C_in, Kd, Kh, Kw],
    stride=1, padding=0, dilation=1, groups=1.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in == C_in_w, "Inconsistent in_channels between input and weight."
    assert bias.numel() == C_out

    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    DHW_out = D_out * H_out * W_out
    P = N * DHW_out

    BLOCK_M = 32
    BLOCK_N = 16

    grid = (
        triton.cdiv(P, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )

    conv3d_ncdhw_kernel[grid](
        x, weight, bias, y,
        N, C_in,
        D_in, H_in, W_in,
        C_out,
        K_D, K_H, K_W,
        D_out, H_out, W_out,
        DHW_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y


def instancenorm_clamp_mul_triton(
    x: torch.Tensor,
    multiplier: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    x: [N, C, D, H, W]
    multiplier: [C, 1, 1, 1]
    Applies:
      x1 = x * multiplier
      x2 = InstanceNorm3d(x1) (no affine, eps)
      x3 = clamp(x2, clamp_min, clamp_max)
      y  = x3 * multiplier
    """
    assert x.is_cuda and multiplier.is_cuda
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    N, C, D, H, W = x.shape
    assert multiplier.shape[0] == C

    y = torch.empty_like(x)

    S = D * H * W
    HW = H * W

    BLOCK_S = 128
    grid = (N * C,)

    instancenorm_clamp_mul_kernel[grid](
        x, multiplier, y,
        N, C, D, H, W,
        S, HW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        multiplier.stride(0), multiplier.stride(1), multiplier.stride(2), multiplier.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        eps, clamp_min, clamp_max,
        BLOCK_S=BLOCK_S,
    )
    return y


def channel_max_triton(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, D, H, W]
    returns: [N, D, H, W] = max over channel dimension
    """
    assert x.is_cuda
    x = x.contiguous()

    N, C, D, H, W = x.shape
    y = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)

    P = N * D * H * W
    DHW = D * H * W
    HW = H * W

    BLOCK_P = 128
    grid = (triton.cdiv(P, BLOCK_P),)

    channel_max_kernel[grid](
        x, y,
        N, C, D, H, W,
        P, DHW, HW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_P=BLOCK_P,
    )
    return y


def fused_conv3d_instancenorm_clamp_max(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Full fusion of the original PyTorch forward:
      x = conv3d(x, weight, bias)
      x = x * multiplier
      x = InstanceNorm3d(x)
      x = clamp(x, clamp_min, clamp_max)
      x = x * multiplier
      x = max(x, dim=1)
    """
    y = conv3d_triton(x, weight, bias)
    y = instancenorm_clamp_mul_triton(y, multiplier, clamp_min, clamp_max, eps)
    y = channel_max_triton(y)
    return y


class ModelNew(nn.Module):
    """
    Triton implementation of:
      Conv3d -> *multiplier -> InstanceNorm3d -> clamp -> *multiplier -> max over channel
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Assume kernel_size is an int (as in the original get_init_inputs)
        k = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, k, k, k))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(*multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_conv3d_instancenorm_clamp_max(
            x,
            self.weight,
            self.bias,
            self.multiplier,
            self.clamp_min,
            self.clamp_max,
            self.eps,
        )
