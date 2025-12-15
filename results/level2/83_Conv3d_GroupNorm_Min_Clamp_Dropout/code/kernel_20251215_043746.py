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
):
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

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution: stride=1, padding=0, dilation=1 (as in nn.Conv3d defaults)
    for c in range(0, C_in):
        for kd in range(0, K_D):
            for kh in range(0, K_H):
                for kw in range(0, K_W):
                    id_idx = od_idx + kd
                    ih_idx = oh_idx + kh
                    iw_idx = ow_idx + kw

                    x_ptrs = (
                        x_ptr
                        + n_idx[:, None] * stride_xn
                        + c * stride_xc
                        + id_idx[:, None] * stride_xd
                        + ih_idx[:, None] * stride_xh
                        + iw_idx[:, None] * stride_xw
                    )
                    mask_x = (
                        mask_m[:, None]
                        & (n_idx[:, None] < N)
                        & (id_idx[:, None] < D_in)
                        & (ih_idx[:, None] < H_in)
                        & (iw_idx[:, None] < W_in)
                    )
                    x = tl.load(x_ptrs, mask=mask_x, other=0.0)

                    w_ptrs = (
                        w_ptr
                        + offs_n[None, :] * stride_wo
                        + c * stride_wc
                        + kd * stride_wd
                        + kh * stride_wh
                        + kw * stride_ww
                    )
                    w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)

                    acc += x * w  # x: [M,1], w: [1,N] -> broadcast to [M,N]

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    mask_yn = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_yn)


@triton.jit
def groupnorm_min_clamp_kernel(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    N, C, D, H, W,
    groups, group_size,
    eps, min_value, max_value,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    ngroups_total = N * groups

    # Guard against over-launch
    if pid >= ngroups_total:
        return

    n = pid // groups
    g = pid % groups

    numel = group_size * D * H * W
    numel_f = tl.full((), numel, tl.float32)

    # First pass: compute mean and variance over the group
    sum_val = 0.0
    sum_sq = 0.0

    for ci in range(0, group_size):
        c = g * group_size + ci
        for d in range(0, D):
            for h in range(0, H):
                for w0 in range(0, W, BLOCK_W):
                    offs_w = w0 + tl.arange(0, BLOCK_W)
                    mask_w = offs_w < W

                    x_ptrs = (
                        x_ptr
                        + n * stride_xn
                        + c * stride_xc
                        + d * stride_xd
                        + h * stride_xh
                        + offs_w * stride_xw
                    )
                    x = tl.load(x_ptrs, mask=mask_w, other=0.0)
                    sum_val += tl.sum(x, axis=0)
                    sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / numel_f
    var = sum_sq / numel_f - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, min, clamp
    for ci in range(0, group_size):
        c = g * group_size + ci
        gamma = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)

        for d in range(0, D):
            for h in range(0, H):
                for w0 in range(0, W, BLOCK_W):
                    offs_w = w0 + tl.arange(0, BLOCK_W)
                    mask_w = offs_w < W

                    x_ptrs = (
                        x_ptr
                        + n * stride_xn
                        + c * stride_xc
                        + d * stride_xd
                        + h * stride_xh
                        + offs_w * stride_xw
                    )
                    x = tl.load(x_ptrs, mask=mask_w, other=0.0)

                    x_hat = (x - mean) * rstd
                    y = x_hat * gamma + beta

                    # x = torch.min(x, min_value)
                    y = tl.minimum(y, min_value)
                    # x = torch.clamp(x, min=min_value, max=max_value)
                    y = tl.maximum(y, min_value)
                    y = tl.minimum(y, max_value)

                    y_ptrs = (
                        y_ptr
                        + n * stride_xn
                        + c * stride_xc
                        + d * stride_xd
                        + h * stride_xh
                        + offs_w * stride_xw
                    )
                    tl.store(y_ptrs, y, mask=mask_w)


def conv3d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    3D convolution (N, C_in, D, H, W) with weight (C_out, C_in, K_D, K_H, K_W),
    stride=1, padding=0, dilation=1. Bias is (C_out,).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in_w == C_in

    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

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
        BLOCK_M=32, BLOCK_N=16,
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

    y = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(N * groups, 1),)

    groupnorm_min_clamp_kernel[grid](
        x, y,
        weight, bias,
        N, C, D, H, W,
        groups, group_size,
        float(eps), float(min_value), float(max_value),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_W=32,
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
