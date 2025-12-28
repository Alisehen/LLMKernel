import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
    ],
    key=["W_out", "C_in", "C_out"],
)
@triton.jit
def pooled_conv_transpose3d_kernel(
    x_ptr,           # *const T: [N, C_in, D_in, H_in, W_in]
    w_ptr,           # *const T: [C_in, C_out, 3, 3, 3]
    bias_eff_ptr,    # *const T: [C_out]
    y_ptr,           # *mut   T: [N, C_out, D_out, H_out, W_out]
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    factor_conv,     # scale1 * scale2 / 8.0 (float32)
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel computing:

        y = ((ConvTranspose3d(x, w, b_conv) * scale1)
             .avg_pool3d(kernel=2, stride=2)
             + bias_outer) * scale2

    algebraically as:

        y = factor_conv * pooled_conv(x, w) + bias_eff

    where:
        factor_conv = scale1 * scale2 / 8
        bias_eff[c] = scale2 * (scale1 * b_conv[c] + bias_outer[c])

    Fast path specialization:
        - kernel_size    = (3, 3, 3)
        - stride         = (2, 2, 2)
        - padding        = (1, 1, 1)
        - dilation       = (1, 1, 1)
        - output_padding = (0, 0, 0)
        - groups         = 1

    Grid layout (1D flattening):
        grid_w    = ceil_div(W_out, BLOCK_W)
        total_pid = grid_w * (N * C_out * D_out * H_out)

        pid = program_id(0) in [0, total_pid)
        pid_w   = pid % grid_w                  -> tile along W_out
        pid_ncdh = pid // grid_w                -> linear over (N, C_out, D_out, H_out)
    """
    pid = tl.program_id(0)

    # Compute number of tiles along W_out given BLOCK_W
    grid_w = (W_out + BLOCK_W - 1) // BLOCK_W

    # Decode width tile and linear (n, c_out, d_out, h_out) index
    pid_w = pid % grid_w
    pid_ncdh = pid // grid_w

    # Width offsets for this program
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_o = offs_w < W_out
    tl.multiple_of(offs_w, BLOCK_W)

    # Decode (n, c_out, d_out, h_out) from pid_ncdh
    h_out = pid_ncdh % H_out
    tmp = pid_ncdh // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c_out = tmp % C_out
    n = tmp // C_out

    # Initialize accumulator in fp32 for numerical stability
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Load effective bias for this output channel (scalar)
    bias_val = tl.load(bias_eff_ptr + c_out)

    # ------------------------------------------------------------------
    # Main compute: fused pooled ConvTranspose3d
    # ConvTranspose3d (3x3x3, stride=2, padding=1) followed by AvgPool3d
    # ------------------------------------------------------------------
    ci = 0
    while ci < C_in:
        # Base offsets for this (n, ci) pair
        x_nc_base = n * stride_xn + ci * stride_xc
        w_nc_base = ci * stride_wn + c_out * stride_wc

        # Depth candidates (3 positions)
        for ad in range(3):
            if ad == 0:
                kz = 1
                iz = d_out
            elif ad == 1:
                kz = 0
                iz = d_out + 1
            else:
                kz = 2
                iz = d_out

            x_ncd_base = x_nc_base + iz * stride_xd
            w_ncd_base = w_nc_base + kz * stride_wd

            # Height candidates (3 positions)
            for ah in range(3):
                if ah == 0:
                    ky = 1
                    iy = h_out
                elif ah == 1:
                    ky = 0
                    iy = h_out + 1
                else:
                    ky = 2
                    iy = h_out

                x_ncdh_base = x_ncd_base + iy * stride_xh
                w_ncdh_base = w_ncd_base + ky * stride_wh

                # Width candidates (3 positions, share same offs_w & mask_o)
                for aw in range(3):
                    if aw == 0:
                        kx = 1
                        ix = offs_w
                    elif aw == 1:
                        kx = 0
                        ix = offs_w + 1
                    else:
                        kx = 2
                        ix = offs_w

                    # Load input slice and weight, FMA into accumulator
                    x_ptrs = x_ptr + x_ncdh_base + ix * stride_xw
                    x_vals = tl.load(x_ptrs, mask=mask_o, other=0.0)
                    w_val = tl.load(w_ptr + w_ncdh_base + kx * stride_ww)

                    acc += x_vals * w_val

        ci += 1

    # Fuse scaling and bias
    acc = acc * factor_conv + bias_val

    # Store results (cast back to output dtype implicitly)
    y_base = (
        n * stride_yn
        + c_out * stride_yc
        + d_out * stride_yd
        + h_out * stride_yh
    )
    y_ptrs = y_ptr + y_base + offs_w * stride_yw
    tl.store(y_ptrs, acc, mask=mask_o)


def fused_pooled_conv_transpose3d(x, conv, scale1, bias_outer, scale2):
    """
    Fuses:
        x = conv_transpose3d(x, weight, bias_conv)
        x = x * scale1
        x = AvgPool3d(kernel=2, stride=2)(x)
        x = x + bias_outer  # (C_out, 1, 1, 1)
        x = x * scale2

    into a single Triton kernel that directly computes the pooled
    transposed-convolution output without materializing the large
    intermediate volume.

    Fast path is specialized for:
        kernel_size    = (3, 3, 3)
        stride         = (2, 2, 2)
        padding        = (1, 1, 1)
        dilation       = (1, 1, 1)
        output_padding = (0, 0, 0)
        groups         = 1
    """

    # Fallback for non-CUDA or unsupported dtypes
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.float32, torch.bfloat16)):
        y = conv(x)
        y = y * scale1
        y = torch.nn.functional.avg_pool3d(y, kernel_size=2, stride=2)
        y = y + bias_outer
        y = y * scale2
        return y

    # Ensure contiguous tensors
    x = x.contiguous()
    w = conv.weight.contiguous()
    bias_outer = bias_outer.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, kD, kH, kW = w.shape

    # Extract conv parameters
    stride_d, stride_h, stride_w_ = conv.stride
    pad_d, pad_h, pad_w_ = conv.padding
    dil_d, dil_h, dil_w_ = conv.dilation
    op_d, op_h, op_w = conv.output_padding
    groups = conv.groups

    # Check fast-path configuration
    if not (
        (C_in == C_in_w)
        and (groups == 1)
        and (kD, kH, kW) == (3, 3, 3)
        and (stride_d, stride_h, stride_w_) == (2, 2, 2)
        and (pad_d, pad_h, pad_w_) == (1, 1, 1)
        and (dil_d, dil_h, dil_w_) == (1, 1, 1)
        and (op_d, op_h, op_w) == (0, 0, 0)
    ):
        y = conv(x)
        y = y * scale1
        y = torch.nn.functional.avg_pool3d(y, kernel_size=2, stride=2)
        y = y + bias_outer
        y = y * scale2
        return y

    # ConvTranspose3d output dims:
    #   out_t = (in - 1) * stride - 2 * pad + kernel + output_padding
    D_t = (D_in - 1) * stride_d - 2 * pad_d + kD + op_d
    H_t = (H_in - 1) * stride_h - 2 * pad_h + kH + op_h
    W_t = (W_in - 1) * stride_w_ - 2 * pad_w_ + kW + op_w

    # AvgPool3d(kernel=2, stride=2) output dims
    D_out = D_t // 2
    H_out = H_t // 2
    W_out = W_t // 2

    # Degenerate case: pooled output has zero-size in any spatial dim
    if D_out == 0 or H_out == 0 or W_out == 0:
        y = conv(x)
        y = y * scale1
        y = torch.nn.functional.avg_pool3d(y, kernel_size=2, stride=2)
        y = y + bias_outer
        y = y * scale2
        return y

    # Effective bias:
    #   bias_eff[c] = scale2 * (scale1 * b_conv[c] + bias_outer[c])
    if conv.bias is not None:
        bias_conv = conv.bias
    else:
        bias_conv = torch.zeros(C_out, device=x.device, dtype=x.dtype)

    bias_outer_flat = bias_outer.view(-1)
    bias_eff = (bias_conv * scale1 + bias_outer_flat) * scale2
    bias_eff = bias_eff.contiguous()

    # Output tensor
    y = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    # Strides
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww = w.stride()
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw = y.stride()

    # Combined scaling factor for conv contributions
    factor_conv = float(scale1 * scale2 / 8.0)

    # Launch configuration via autotuned BLOCK_W
    def grid(meta):
        block_w = meta["BLOCK_W"]
        grid_w = triton.cdiv(W_out, block_w)
        grid_ncdh = N * C_out * D_out * H_out
        grid0 = grid_w * grid_ncdh
        return (max(1, grid0),)

    # Launch Triton kernel
    pooled_conv_transpose3d_kernel[grid](
        x, w, bias_eff, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
        stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
        factor_conv,
    )

    return y


class ModelNew(nn.Module):
    """
    Optimized model that replaces:

        x = ConvTranspose3d(...)
        x = x * scale1
        x = AvgPool3d(kernel_size=2, stride=2)(x)
        x = x + bias
        x = x * scale2

    with a single fused Triton kernel that directly computes the
    pooled transposed-convolution output for the common configuration:

        kernel_size    = 3
        stride         = 2
        padding        = 1
        dilation       = 1
        output_padding = 0
        groups         = 1

    For other configurations or CPU tensors, it falls back to the
    exact PyTorch reference computation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale1,
        scale2,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))

    def forward(self, x):
        if x.is_cuda:
            return fused_pooled_conv_transpose3d(
                x,
                self.conv_transpose,
                self.scale1,
                self.bias,
                self.scale2,
            )
        else:
            x = self.conv_transpose(x)
            x = x * self.scale1
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            x = x + self.bias
            x = x * self.scale2
            return x
