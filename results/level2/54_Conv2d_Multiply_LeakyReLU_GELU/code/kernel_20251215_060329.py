import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_mul_lrelu_gelu_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, y_ptr,
    N, C_OUT, H_in, W_in, H_out, W_out, P,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    negative_slope, inv_sqrt2,
    C_IN: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_OUT

    HW_out = H_out * W_out

    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ic in range(C_IN):
        for kh in range(K_H):
            for kw in range(K_W):
                x_h = oh_idx + kh
                x_w = ow_idx + kw

                x_ptrs = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + ic * stride_xc
                    + x_h[:, None] * stride_xh
                    + x_w[:, None] * stride_xw
                )
                w_ptrs = (
                    w_ptr
                    + offs_n[None, :] * stride_wo
                    + ic * stride_wc
                    + kh * stride_wh
                    + kw * stride_ww
                )

                x_vals = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
                w_vals = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)

                acc += x_vals * w_vals

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Multiply by per-channel multiplier
    mult_vals = tl.load(mult_ptr + offs_n, mask=mask_n, other=1.0)
    acc *= mult_vals[None, :]

    # LeakyReLU
    acc = tl.where(acc >= 0.0, acc, acc * negative_slope)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    t = acc * inv_sqrt2
    erf_t = tl.math.erf(t)
    acc = 0.5 * acc * (1.0 + erf_t)

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )

    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_conv2d_mul_lrelu_gelu(x, weight, bias, multiplier, negative_slope=0.01):
    """
    x: [N, C_in, H_in, W_in]
    weight: [C_out, C_in, K_h, K_w]
    bias: [C_out]
    multiplier: [C_out] or [C_out, 1, 1]
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda and bias.is_cuda and multiplier.is_cuda
    assert x.dtype == torch.float32, "Kernel currently assumes float32"

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels must match"
    assert K_h == K_w, "Kernel assumed square; got K_h != K_w"

    # Only support stride=1, padding=0, dilation=1
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1
    assert H_out > 0 and W_out > 0, "Invalid output size"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten multiplier to [C_out]
    if multiplier.dim() == 3:
        mult_flat = multiplier.view(-1)
    else:
        mult_flat = multiplier
    assert mult_flat.numel() == C_out

    P = N * H_out * W_out

    x_strides = x.stride()
    w_strides = weight.stride()
    y_strides = y.stride()

    grid = lambda META: (
        triton.cdiv(P, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_mul_lrelu_gelu_kernel[grid](
        x, weight, bias, mult_flat, y,
        N, C_out, H_in, W_in, H_out, W_out, P,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        w_strides[0], w_strides[1], w_strides[2], w_strides[3],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3],
        negative_slope, 1.0 / math.sqrt(2.0),
        C_IN=C_in, K_H=K_h, K_W=K_w,
        BLOCK_M=32, BLOCK_N=32,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-fused version of:
      Conv2d -> channel-wise multiply -> LeakyReLU -> GELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # Match default nn.LeakyReLU()
        self.negative_slope = 0.01

    def forward(self, x):
        return fused_conv2d_mul_lrelu_gelu(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )
