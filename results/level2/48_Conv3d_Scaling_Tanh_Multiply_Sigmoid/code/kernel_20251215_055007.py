import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small, register-friendly baseline
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider in OC
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64},
            num_warps=4,
            num_stages=2,
        ),
        # Wider in spatial / batch dimension
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['P', 'OC'],
)
@triton.jit
def conv3d_fused_tanh_sigmoid_kernel(
    x_ptr,            # float32[N, C_in, D_in, H_in, W_in]
    w_ptr,            # float32[OC, C_in, KD, KH, KW]
    conv_bias_ptr,    # float32[OC]
    scale_ptr,        # float32[OC, 1, 1, 1]
    bias2_ptr,        # float32[OC, 1, 1, 1]
    y_ptr,            # float32[N, OC, D_out, H_out, W_out]

    N,
    D_in, H_in, W_in,
    OC,
    D_out, H_out, W_out,
    P,                # N * D_out * H_out * W_out

    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    scale_stride_c, bias2_stride_c,

    C_IN: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids for 2D tiling over:
    #   M = P  (flattened [N, D_out, H_out, W_out])
    #   N = OC (output channels)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < OC
    mask = mask_m[:, None] & mask_n[None, :]

    # Decode flattened M-index -> (n_idx, od_idx, oh_idx, ow_idx)
    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    od_idx = rem // HW_out
    rem2 = rem % HW_out
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    # Base pointer offsets for input corresponding to (n, od, oh, ow)
    x_base = (
        n_idx * stride_xn
        + od_idx * stride_xd
        + oh_idx * stride_xh
        + ow_idx * stride_xw
    )

    # Base pointer for weights for this OC tile
    w_oc_base = w_ptr + offs_n * stride_wn

    # Accumulator for convolution result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Convolution reduction over C_IN * KD * KH * KW
    # Fully unrolled static loops for maximum FMAD throughput.
    # -------------------------------------------------------------------------
    for ic in tl.static_range(0, C_IN):
        ic_offset_x = ic * stride_xc
        ic_offset_w = ic * stride_wc

        for kd in tl.static_range(0, KD):
            kd_offset_x = kd * stride_xd
            kd_offset_w = kd * stride_wd

            for kh in tl.static_range(0, KH):
                kh_offset_x = kh * stride_xh
                kh_offset_w = kh * stride_wh

                for kw in tl.static_range(0, KW):
                    kw_offset_x = kw * stride_xw
                    kw_offset_w = kw * stride_ww

                    # X values: [BLOCK_M]
                    x_ptrs = (
                        x_ptr
                        + x_base
                        + ic_offset_x
                        + kd_offset_x
                        + kh_offset_x
                        + kw_offset_x
                    )
                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)

                    # W values: [BLOCK_N] for current (ic, kd, kh, kw, offs_n)
                    w_ptrs = (
                        w_oc_base
                        + ic_offset_w
                        + kd_offset_w
                        + kh_offset_w
                        + kw_offset_w
                    )
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)

                    # Outer product accumulate
                    acc += x_vals[:, None] * w_vals[None, :]

    # -------------------------------------------------------------------------
    # Fused post-convolution ops with aggressive in-place updates
    # to minimize register live ranges:
    #   + conv bias (per-channel)
    #   * scaling_factor (per-channel)
    #   tanh
    #   * bias2 (per-channel)
    #   sigmoid
    # -------------------------------------------------------------------------

    # Add convolution bias: acc += b[oc]
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += conv_b[None, :]

    # Multiply by per-channel scale: acc *= scale[oc]
    scale = tl.load(scale_ptr + offs_n * scale_stride_c, mask=mask_n, other=0.0)
    acc *= scale[None, :]

    # Tanh activation (in-place on acc to reduce temporaries):
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    acc *= 2.0
    tmp = tl.exp(acc)
    acc = (tmp - 1.0) / (tmp + 1.0)

    # Multiply by bias2: acc *= bias2[oc]
    bias2 = tl.load(bias2_ptr + offs_n * bias2_stride_c, mask=mask_n, other=0.0)
    acc *= bias2[None, :]

    # Sigmoid (in-place):
    # σ(z) = 1 / (1 + e^{-z})
    acc = -acc
    tmp = tl.exp(acc)
    acc = 1.0 / (1.0 + tmp)

    # -------------------------------------------------------------------------
    # Store result back to y[n, oc, od, oh, ow]
    # -------------------------------------------------------------------------
    y_base = (
        n_idx * stride_yn
        + od_idx * stride_yd
        + oh_idx * stride_yh
        + ow_idx * stride_yw
    )
    y_ptrs = y_ptr + y_base[:, None] + offs_n[None, :] * stride_yc
    tl.store(y_ptrs, acc, mask=mask)


def fused_conv3d_scaled_tanh_sigmoid(x, weight, conv_bias, scale, bias2):
    """
    x:        [N, C_in, D_in, H_in, W_in]       (float32, CUDA)
    weight:   [OC, C_in, KD, KH, KW]           (float32, CUDA, contiguous)
    conv_bias:[OC]                             (float32, CUDA)
    scale:    [OC, 1, 1, 1]                    (float32, CUDA)
    bias2:    [OC, 1, 1, 1]                    (float32, CUDA)
    """
    assert x.is_cuda
    assert weight.is_cuda and conv_bias.is_cuda and scale.is_cuda and bias2.is_cuda

    N, C_in, D_in, H_in, W_in = x.shape
    OC, C_in_w, KD, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight."

    # Stride=1, padding=0, dilation=1
    D_out = D_in - KD + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    assert D_out > 0 and H_out > 0 and W_out > 0

    y = torch.empty((N, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * D_out * H_out * W_out

    # Grid depends on BLOCK_M / BLOCK_N chosen by autotuner
    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

    conv3d_fused_tanh_sigmoid_kernel[grid](
        x,
        weight,
        conv_bias,
        scale,
        bias2,
        y,
        N,
        D_in, H_in, W_in,
        OC,
        D_out, H_out, W_out,
        P,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        scale.stride(0), bias2.stride(0),
        C_IN=C_in,
        KD=KD,
        KH=KH,
        KW=KW,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-based implementation of:
        Conv3d -> * scaling_factor -> tanh -> * bias -> sigmoid
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        # nn.Conv3d holds weights/bias with same semantics as original model
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # Per-channel scaling_factor and bias
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Ensure parameters are on same device as x
        self.conv.to(x.device)
        self.scaling_factor.data = self.scaling_factor.data.to(x.device)
        self.bias.data = self.bias.data.to(x.device)

        return fused_conv3d_scaled_tanh_sigmoid(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )
