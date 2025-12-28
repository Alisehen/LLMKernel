import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def deconv_mul_global_avg2d_kernel(
    x_ptr,            # float*  [B, Cin, Hin, Win]
    w_ptr,            # float*  [Cin, Cout, kH, kW]
    bias_ptr,         # float*  [Cout] or nullptr
    y_ptr,            # float*  [B, Cout, 1, 1]

    B, Cin, Hin, Win, Cout,
    stride_h, stride_w,
    pad_h, pad_w,
    Hout, Wout,

    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wci, stride_wco, stride_wkh, stride_wkw,
    stride_bias_c,
    stride_yb, stride_yc,

    scale,           # multiplier / (Hout * Wout)
    multiplier,      # multiplier (for bias term)

    HAS_BIAS: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Fused kernel that computes:

        y[b, oc, 0, 0] = multiplier * mean_{oh,ow}(
                             conv_transpose2d(x, w, bias)[b, oc, oh, ow]
                         )

    without ever materializing the full (Hout, Wout) output.

    It accumulates the spatial sum of the transposed-convolution output
    for each (b, oc), then applies scaling and bias analytically.
    """
    pid_b = tl.program_id(0)
    pid_oc_block = tl.program_id(1)

    # Batch index for this program
    b = pid_b
    b_mask = b < B

    # Output-channel tile
    oc_start = pid_oc_block * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = (oc_offsets < Cout) & b_mask

    # Accumulator over spatial positions (sum over Hout * Wout of the conv output only, no bias)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and spatial positions
    for ic in range(0, Cin):
        for ih in range(0, Hin):
            oh_base = ih * stride_h - pad_h
            for iw in range(0, Win):
                ow_base = iw * stride_w - pad_w

                # Load input x[b, ic, ih, iw]
                x_ptr_elt = (
                    x_ptr
                    + b * stride_xb
                    + ic * stride_xc
                    + ih * stride_xh
                    + iw * stride_xw
                )
                x_val = tl.load(x_ptr_elt, mask=b_mask, other=0.0)
                x_val_f32 = x_val.to(tl.float32)

                # Skip if batch is out-of-range
                # (masking through b_mask already covers this; no extra branch needed)

                # Loop over kernel positions
                for kh in range(0, K_H):
                    oh = oh_base + kh
                    mask_h = (oh >= 0) & (oh < Hout)

                    for kw in range(0, K_W):
                        ow = ow_base + kw
                        mask_w = (ow >= 0) & (ow < Wout)

                        # This (ih, iw, kh, kw) contributes exactly once if inside output bounds
                        mask_hw = mask_h & mask_w
                        contrib_mask = oc_mask & mask_hw

                        # Load weights w[ic, oc, kh, kw] as a vector over oc_offsets
                        w_ptrs = (
                            w_ptr
                            + ic * stride_wci
                            + oc_offsets * stride_wco
                            + kh * stride_wkh
                            + kw * stride_wkw
                        )
                        w_vals = tl.load(w_ptrs, mask=contrib_mask, other=0.0)

                        # Accumulate contribution
                        acc += x_val_f32 * w_vals

    # Apply global averaging (over Hout * Wout) and multiplier for convolution part
    out_vals = acc * scale  # scale = multiplier / (Hout * Wout) for conv term

    # Add bias contribution: multiplier * bias[oc]
    if HAS_BIAS:
        bias_vals = tl.load(
            bias_ptr + oc_offsets * stride_bias_c,
            mask=oc_mask,
            other=0.0,
        )
        bias_vals_f32 = bias_vals.to(tl.float32)
        out_vals += multiplier * bias_vals_f32

    # Store result y[b, oc, 0, 0]
    y_ptrs = y_ptr + b * stride_yb + oc_offsets * stride_yc
    tl.store(y_ptrs, out_vals, mask=oc_mask)


def fused_conv_transpose_mul_global_avg2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    multiplier: float,
) -> torch.Tensor:
    """
    Fused PyTorch-equivalent of:

        y_full = F.conv_transpose2d(x, weight, bias, stride, padding, output_padding)
        y_full = y_full * multiplier
        y = torch.mean(y_full, dim=[2, 3], keepdim=True)
        y = torch.mean(y, dim=[2, 3], keepdim=True)  # no-op after keepdim

    Returns: y with shape [B, Cout, 1, 1]
    """
    assert x.ndim == 4, "Expected input of shape [B, Cin, Hin, Win]"
    B, Cin, Hin, Win = x.shape
    Cin_w, Cout, kH, kW = weight.shape
    assert Cin_w == Cin, "Inconsistent in_channels between input and weight"

    # Only groups=1, dilation=1 supported in this fused kernel
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(output_padding, int):
        op_h = op_w = output_padding
    else:
        op_h, op_w = output_padding

    # Compute output spatial size exactly as ConvTranspose2d does (dilation=1)
    Hout = (Hin - 1) * stride_h - 2 * pad_h + (kH - 1) * 1 + op_h + 1
    Wout = (Win - 1) * stride_w - 2 * pad_w + (kW - 1) * 1 + op_w + 1

    device = x.device
    dtype = x.dtype

    # Output is compact [B, Cout, 1, 1]
    y = torch.empty((B, Cout, 1, 1), device=device, dtype=dtype)

    # Combined scaling for convolution part: multiplier / (Hout * Wout)
    scale = float(multiplier) / float(Hout * Wout)

    # Strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wci, stride_wco, stride_wkh, stride_wkw = weight.stride()
    if bias is not None:
        stride_bias_c = bias.stride(0)
        has_bias = True
        bias_ptr = bias
    else:
        # Dummy tensor to satisfy ptr argument (won't be used when HAS_BIAS=False)
        bias_ptr = weight.new_empty(1)
        stride_bias_c = 0
        has_bias = False

    stride_yb, stride_yc, _, _ = y.stride()

    BLOCK_OC = 128

    grid = lambda META: (
        B,
        triton.cdiv(Cout, META["BLOCK_OC"]),
    )

    deconv_mul_global_avg2d_kernel[grid](
        x,
        weight,
        bias_ptr,
        y,
        B,
        Cin,
        Hin,
        Win,
        Cout,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        Hout,
        Wout,
        stride_xb,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_wci,
        stride_wco,
        stride_wkh,
        stride_wkw,
        stride_bias_c,
        stride_yb,
        stride_yc,
        scale,
        float(multiplier),
        HAS_BIAS=has_bias,
        K_H=kH,
        K_W=kW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model that fuses:

      ConvTranspose2d + scalar multiplication + two global average poolings

    into a single Triton kernel which directly computes a (B, Cout, 1, 1) tensor
    without materializing the full (Hout, Wout) feature map.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.multiplier = float(multiplier)

    def forward(self, x):
        # Use Triton fused kernel instead of cuDNN conv_transpose2d + pooling
        w = self.conv_transpose.weight
        b = self.conv_transpose.bias
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        output_padding = self.conv_transpose.output_padding

        # groups and dilation must match what the fused kernel assumes
        assert self.conv_transpose.groups == 1, "Fused kernel only supports groups=1"
        if isinstance(self.conv_transpose.dilation, tuple):
            assert self.conv_transpose.dilation == (1, 1), "dilation != 1 not supported"
        else:
            assert self.conv_transpose.dilation == 1, "dilation != 1 not supported"

        return fused_conv_transpose_mul_global_avg2d(
            x,
            w,
            b,
            stride,
            padding,
            output_padding,
            self.multiplier,
        )
