# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def pooled_conv_transpose3d_kernel(
    x_ptr,           # *const T: [N, C_in, D_in, H_in, W_in]
    w_ptr,           # *const T: [C_in, C_out, 3, 3, 3]
    bias_eff_ptr,    # *const T: [C_out]
    y_ptr,           # *mut   T: [N, C_out, D_out, H_out, W_out]
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    n_w_blocks,      # number of W-out blocks = ceil(W_out / BLOCK_W)
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    factor_conv,     # scale1 * scale2 / 8.0
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel computing:

        y = ((ConvTranspose3d(x, w, b_conv) * scale1)
             .avg_pool3d(kernel=2, stride=2)
             + bias_outer) * scale2

    but algebraically fused into:

        y = factor_conv * pooled_conv(x, w) + bias_eff

    where:
        factor_conv = scale1 * scale2 / 8
        bias_eff[c] = scale2 * (scale1 * b_conv[c] + bias_outer[c])

    Assumptions (enforced in Python wrapper):
        - kernel_size = (3, 3, 3)
        - stride      = (2, 2, 2)
        - padding     = (1, 1, 1)
        - dilation    = (1, 1, 1)
        - output_padding = (0, 0, 0)
        - groups = 1
    """

    pid = tl.program_id(0)

    # Decode program id into (n, c_out, d_out, h_out, w_block)
    w_block_id = pid % n_w_blocks
    tmp = pid // n_w_blocks

    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c_out = tmp % C_out
    n = tmp // C_out

    # W indices handled by this program
    w_start = w_block_id * BLOCK_W
    w_idx = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_idx < W_out

    # Initialize accumulator for this (n, c_out, d_out, h_out, w_idx) tile
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Preload effective bias for this output channel, broadcast to vector
    bias_val = tl.load(bias_eff_ptr + c_out)
    bias_vec = bias_val

    # Candidate (input index, kernel index) per axis for stride=2, pad=1, k=3:
    # For pooled index p, conv-transpose output indices are 2p and 2p+1.
    # Inverting j = i*2 - 1 + k gives three (i,k) pairs per axis:
    #   (i=p,   k=1) from j=2p
    #   (i=p+1, k=0) from j=2p+1
    #   (i=p,   k=2) from j=2p+1
    #
    # We'll loop ad, ah, aw in {0,1,2} selecting these pairs.

    # Loop over input channels
    ci = 0
    while ci < C_in:
        # Loop over depth-axis candidate (i_z, k_z)
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

            # Check depth bounds (scalar)
            valid_d = (iz >= 0) & (iz < D_in)
            iz_eff = tl.where(valid_d, iz, 0)

            # Loop over height-axis candidate (i_y, k_y)
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

                # Check height bounds (scalar)
                valid_h = (iy >= 0) & (iy < H_in)
                iy_eff = tl.where(valid_h, iy, 0)

                # Combined scalar validity so far (to avoid needless work)
                valid_dh = valid_d & valid_h

                # Loop over width-axis candidate (i_x, k_x)
                for aw in range(3):
                    if aw == 0:
                        kx = 1
                        ix_base_offset = 0  # ix = w_idx + 0
                    elif aw == 1:
                        kx = 0
                        ix_base_offset = 1  # ix = w_idx + 1
                    else:
                        kx = 2
                        ix_base_offset = 0  # ix = w_idx + 0

                    # ix is vector over BLOCK_W
                    ix = w_idx + ix_base_offset

                    # Check width bounds (vector) and accumulate full mask
                    valid_w = (ix >= 0) & (ix < W_in) & mask_w
                    # Broadcast depth/height validity to vector
                    mask = valid_w & valid_dh

                    # Clamp indices to within-range values for safe loads
                    ix_eff = tl.where(mask, ix, 0)

                    # Compute base offset for x (without width)
                    x_base = (
                        n * stride_xn
                        + ci * stride_xc
                        + iz_eff * stride_xd
                        + iy_eff * stride_xh
                    )

                    x_ptrs = x_ptr + x_base + ix_eff * stride_xw

                    # Load input slice and weight scalar
                    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

                    w_offset = (
                        ci * stride_wn
                        + c_out * stride_wc
                        + kz * stride_wd
                        + ky * stride_wh
                        + kx * stride_ww
                    )
                    w_val = tl.load(w_ptr + w_offset)

                    # FMA into accumulator
                    acc += x_vals * w_val

        ci += 1

    # Apply combined scaling and add effective bias
    acc = acc * factor_conv + bias_vec

    # Store results
    y_base = (
        n * stride_yn
        + c_out * stride_yc
        + d_out * stride_yd
        + h_out * stride_yh
    )
    y_ptrs = y_ptr + y_base + w_idx * stride_yw
    tl.store(y_ptrs, acc, mask=mask_w)


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

    This optimized path is specialized for:
        kernel_size    = (3, 3, 3)
        stride         = (2, 2, 2)
        padding        = (1, 1, 1)
        dilation       = (1, 1, 1)
        output_padding = (0, 0, 0)
        groups         = 1
    and requires CUDA tensors. For other configurations or CPU, it
    falls back to the PyTorch reference computation.
    """
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.float32, torch.bfloat16)):
        # Fallback: exact PyTorch behavior
        y = conv(x)
        y = y * scale1
        y = F.avg_pool3d(y, kernel_size=2, stride=2)
        y = y + bias_outer
        y = y * scale2
        return y

    # Ensure contiguous tensors
    x = x.contiguous()
    w = conv.weight.contiguous()
    bias_outer = bias_outer.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, kD, kH, kW = w.shape

    # Only handle the targeted configuration in the fast path
    stride_d, stride_h, stride_w = conv.stride
    pad_d, pad_h, pad_w = conv.padding
    dil_d, dil_h, dil_w = conv.dilation
    op_d, op_h, op_w = conv.output_padding
    groups = conv.groups

    if not (
        (C_in == C_in_w)
        and (groups == 1)
        and (kD, kH, kW) == (3, 3, 3)
        and (stride_d, stride_h, stride_w) == (2, 2, 2)
        and (pad_d, pad_h, pad_w) == (1, 1, 1)
        and (dil_d, dil_h, dil_w) == (1, 1, 1)
        and (op_d, op_h, op_w) == (0, 0, 0)
    ):
        # Fallback: non-optimized configuration
        y = conv(x)
        y = y * scale1
        y = F.avg_pool3d(y, kernel_size=2, stride=2)
        y = y + bias_outer
        y = y * scale2
        return y

    # Compute intermediate transposed-conv output dims
    # PyTorch formula: out = (in - 1) * stride - 2 * pad + kernel + output_padding
    D_t = (D_in - 1) * stride_d - 2 * pad_d + kD + op_d
    H_t = (H_in - 1) * stride_h - 2 * pad_h + kH + op_h
    W_t = (W_in - 1) * stride_w - 2 * pad_w + kW + op_w

    # AvgPool3d(kernel=2, stride=2) output dims: floor(out_t / 2)
    D_out = D_t // 2
    H_out = H_t // 2
    W_out = W_t // 2

    # Prepare effective bias:
    #   bias_eff[c] = scale2 * (scale1 * b_conv[c] + bias_outer[c])
    if conv.bias is not None:
        bias_conv = conv.bias
    else:
        bias_conv = torch.zeros(C_out, device=x.device, dtype=x.dtype)

    bias_outer_flat = bias_outer.view(-1)
    bias_eff = (bias_conv * scale1 + bias_outer_flat) * scale2
    bias_eff = bias_eff.contiguous()

    # Output tensor
    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww = w.stride()
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw = y.stride()

    # Combined scaling factor for conv contributions
    factor_conv = float(scale1 * scale2 / 8.0)

    # Launch configuration
    BLOCK_W = 32
    n_w_blocks = (W_out + BLOCK_W - 1) // BLOCK_W
    n_ncdh = N * C_out * D_out * H_out
    grid = lambda META: (max(1, n_ncdh * n_w_blocks),)

    pooled_conv_transpose3d_kernel[grid](
        x, w, bias_eff, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        n_w_blocks,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
        stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
        factor_conv,
        BLOCK_W=BLOCK_W,
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
        # Use fused Triton implementation when possible, else fallback
        if x.is_cuda:
            return fused_pooled_conv_transpose3d(
                x,
                self.conv_transpose,
                self.scale1,
                self.bias,
                self.scale2,
            )
        else:
            # Exact PyTorch reference for CPU
            x = self.conv_transpose(x)
            x = x * self.scale1
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
            x = x + self.bias
            x = x * self.scale2
            return x
