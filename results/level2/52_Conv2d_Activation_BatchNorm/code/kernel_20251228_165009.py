import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_softplus_tanh_mul_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, Kh, Kw,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over N*H_out*W_out
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # reduction tile over C_in*Kh*Kw
):
    pid_m = tl.program_id(0)  # over output spatial+batch (M dimension)
    pid_n = tl.program_id(1)  # over output channels (N dimension)

    M = N * H_out * W_out
    K_tot = C_in * Kh * Kw

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, h_out, w_out)
    hw = H_out * W_out
    n_idx = offs_m // hw
    rem_hw = offs_m % hw
    h_out_idx = rem_hw // W_out
    w_out_idx = rem_hw % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_tot, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_tot

        # Decode reduction index k -> (c_in, kh, kw)
        khw = Kh * Kw
        ci = offs_k // khw
        rem_k = offs_k % khw
        kh_idx = rem_k // Kw
        kw_idx = rem_k % Kw

        # Input coordinates
        h_in = h_out_idx[:, None] + kh_idx[None, :]
        w_in = w_out_idx[:, None] + kw_idx[None, :]

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ci[None, :] * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        a = tl.load(
            x_ptrs,
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0,
        )

        # Weight coordinates: (C_out, C_in, Kh, Kw)
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ci[:, None] * stride_wc
            + kh_idx[:, None] * stride_wh
            + kw_idx[:, None] * stride_ww
        )
        b = tl.load(
            w_ptrs,
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    bias = bias.to(tl.float32)
    acc += bias[None, :]

    # Fused activation: x * tanh(softplus(x))
    # softplus(x) ≈ log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
    t2 = tl.exp(2.0 * sp)
    t = (t2 - 1.0) / (t2 + 1.0)
    acc = t * acc

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + h_out_idx[:, None] * stride_yh
        + w_out_idx[:, None] * stride_yw
    )

    tl.store(
        y_ptrs,
        acc,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


@triton.jit
def conv_softplus_tanh_mul_bn_kernel(
    x_ptr, w_ptr, b_ptr,
    bn_scale_ptr, bn_shift_ptr,
    y_ptr,
    N, C_in, H, W,
    C_out, Kh, Kw,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over N*H_out*W_out
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # reduction tile over C_in*Kh*Kw
):
    pid_m = tl.program_id(0)  # over output spatial+batch (M dimension)
    pid_n = tl.program_id(1)  # over output channels (N dimension)

    M = N * H_out * W_out
    K_tot = C_in * Kh * Kw

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, h_out, w_out)
    hw = H_out * W_out
    n_idx = offs_m // hw
    rem_hw = offs_m % hw
    h_out_idx = rem_hw // W_out
    w_out_idx = rem_hw % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_tot, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_tot

        # Decode reduction index k -> (c_in, kh, kw)
        khw = Kh * Kw
        ci = offs_k // khw
        rem_k = offs_k % khw
        kh_idx = rem_k // Kw
        kw_idx = rem_k % Kw

        # Input coordinates
        h_in = h_out_idx[:, None] + kh_idx[None, :]
        w_in = w_out_idx[:, None] + kw_idx[None, :]

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ci[None, :] * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        a = tl.load(
            x_ptrs,
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0,
        )

        # Weight coordinates: (C_out, C_in, Kh, Kw)
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ci[:, None] * stride_wc
            + kh_idx[:, None] * stride_wh
            + kw_idx[:, None] * stride_ww
        )
        b = tl.load(
            w_ptrs,
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    # Add conv bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    bias = bias.to(tl.float32)
    acc += bias[None, :]

    # Fused activation: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    t2 = tl.exp(2.0 * sp)
    t = (t2 - 1.0) / (t2 + 1.0)
    acc = t * acc

    # Fused BatchNorm (inference): y = acc * scale + shift
    scale = tl.load(bn_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    shift = tl.load(bn_shift_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc * scale[None, :] + shift[None, :]

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + h_out_idx[:, None] * stride_yh
        + w_out_idx[:, None] * stride_yw
    )

    tl.store(
        y_ptrs,
        acc,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


def conv_softplus_tanh_mul_triton(x, weight, bias):
    """
    Fused:
      y = conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
      y = y * tanh(softplus(y))
    Used as the training path (BN applied separately in PyTorch).
    """
    assert x.dim() == 4, "Input must be NCHW"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, Kh, Kw = weight.shape
    assert C_in_w == C_in, "in_channels mismatch between input and weight"
    # Assumes stride=1, padding=0, dilation=1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv_softplus_tanh_mul_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, Kh, Kw,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return y


def conv_softplus_tanh_mul_bn_triton(x, weight, bias, bn_scale, bn_shift):
    """
    Fused inference path:
      z = conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
      z = z * tanh(softplus(z))
      y = BatchNorm(z)   (using running statistics)
    where BatchNorm has been reduced to a per-channel affine: y = z * scale + shift
    """
    assert x.dim() == 4, "Input must be NCHW"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    bn_scale = bn_scale.contiguous()
    bn_shift = bn_shift.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, Kh, Kw = weight.shape
    assert C_in_w == C_in, "in_channels mismatch between input and weight"
    assert bn_scale.numel() == C_out and bn_shift.numel() == C_out, "BN params shape mismatch"

    # Assumes stride=1, padding=0, dilation=1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv_softplus_tanh_mul_bn_kernel[grid](
        x, weight, bias,
        bn_scale, bn_shift,
        y,
        N, C_in, H, W,
        C_out, Kh, Kw,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement of the reference model.

    Training:
        - Conv2d via Triton conv+activation kernel
        - BatchNorm2d via PyTorch (updates running stats, correct gradients)

    Inference (bn.training == False and track_running_stats == True):
        - Fully fused Triton kernel: conv + x*tanh(softplus(x)) + BatchNorm
          using BN running statistics folded into a per-channel affine transform.
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def _get_bn_fused_params(self, x):
        """
        Compute per-channel affine parameters equivalent to BatchNorm inference:
            y = (z - mean) / sqrt(var + eps) * gamma + beta
              = z * (gamma / sqrt(var + eps)) + (beta - mean * gamma / sqrt(var + eps))
        Returns:
            scale, shift  (each of shape [C_out])
        """
        bn = self.bn
        dtype = x.dtype
        device = x.device

        # Running statistics must be present for fused inference
        running_mean = bn.running_mean
        running_var = bn.running_var
        assert running_mean is not None and running_var is not None, \
            "BatchNorm must track running stats for fused inference."

        mean = running_mean
        var = running_var

        if mean.dtype != dtype:
            mean = mean.to(dtype=dtype)
        if var.dtype != dtype:
            var = var.to(dtype=dtype)

        if bn.affine:
            gamma = bn.weight
            beta = bn.bias
            if gamma.dtype != dtype:
                gamma = gamma.to(dtype=dtype)
            if beta.dtype != dtype:
                beta = beta.to(dtype=dtype)
        else:
            num_features = bn.num_features
            gamma = torch.ones(num_features, device=device, dtype=dtype)
            beta = torch.zeros(num_features, device=device, dtype=dtype)

        # scale = gamma / sqrt(var + eps)
        # shift = beta - mean * scale
        scale = gamma / torch.sqrt(var + bn.eps)
        shift = beta - mean * scale

        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        # Inference: fuse BatchNorm into Triton conv+activation kernel
        if (not self.bn.training) and self.bn.track_running_stats:
            bn_scale, bn_shift = self._get_bn_fused_params(x)
            x = conv_softplus_tanh_mul_bn_triton(
                x,
                self.conv.weight,
                self.conv.bias,
                bn_scale,
                bn_shift,
            )
        else:
            # Training (or non-standard BN config): keep BN as separate PyTorch op
            x = conv_softplus_tanh_mul_triton(x, self.conv.weight, self.conv.bias)
            x = self.bn(x)
        return x
