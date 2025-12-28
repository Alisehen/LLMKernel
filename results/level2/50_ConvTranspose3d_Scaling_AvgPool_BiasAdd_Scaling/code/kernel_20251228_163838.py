import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_avgpool3d_fused_kernel(
    x_ptr, w_ptr, conv_b_ptr, bias_after_ptr, y_ptr,
    N, C_in, D_in, H_in, W_in,
    C_out, KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    D_full, H_full, W_full,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wci, stride_wco, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    scale1, scale2,
    BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    """
    Fused kernel implementing:

      ConvTranspose3d -> scale1 -> AvgPool3d(kernel=2,stride=2) -> bias_add -> scale2

    directly from x to the final pooled output, without materializing the
    high-resolution ConvTranspose3d output.
    """
    pid_p = tl.program_id(0)   # pooled spatial tiles
    pid_co = tl.program_id(1)  # output channel tiles
    pid_n = tl.program_id(2)   # batch index

    # Offsets for output channels and pooled spatial positions
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_co = offs_co < C_out
    P_out = D_out * H_out * W_out
    mask_p = offs_p < P_out

    # Decode pooled spatial indices (dp, hp, wp) from linear index offs_p
    hw_out = H_out * W_out
    dp = offs_p // hw_out
    rem = offs_p - dp * hw_out
    hp = rem // W_out
    wp = rem - hp * W_out

    # Base pointers for the current batch element
    x_base = x_ptr + pid_n * stride_xn
    y_base = y_ptr + pid_n * stride_yn

    # Accumulator for sum of conv_transpose outputs over 2x2x2 pooling window
    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    ci_range = tl.arange(0, BLOCK_CI)

    # Loop over input channels in tiles
    for ci_start in range(0, C_in, BLOCK_CI):
        offs_ci = ci_start + ci_range
        ci_mask = offs_ci < C_in

        w_ci_co_mask = ci_mask[:, None] & mask_co[None, :]

        # Loop over transposed-convolution kernel spatial dimensions
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    # Load weights for this (kd,kh,kw) and (ci,co)-tile
                    w_ptrs = (
                        w_ptr
                        + offs_ci[:, None] * stride_wci
                        + offs_co[None, :] * stride_wco
                        + kd * stride_wkd
                        + kh * stride_wkh
                        + kw * stride_wkw
                    )
                    w_vals = tl.load(w_ptrs, mask=w_ci_co_mask, other=0.0)
                    w_t = tl.trans(w_vals)  # (BLOCK_CO, BLOCK_CI)

                    # Accumulate contributions from the 2x2x2 pooling region
                    for dz in range(0, 2):
                        od = dp * 2 + dz
                        t_d = od + pad_d - kd
                        id0 = t_d // stride_d
                        valid_d = (t_d >= 0) & (id0 < D_in) & (id0 * stride_d == t_d)

                        for dy in range(0, 2):
                            oh = hp * 2 + dy
                            t_h = oh + pad_h - kh
                            ih0 = t_h // stride_h
                            valid_h = (t_h >= 0) & (ih0 < H_in) & (ih0 * stride_h == t_h)

                            for dx in range(0, 2):
                                ow = wp * 2 + dx
                                t_w = ow + pad_w - kw
                                iw0 = t_w // stride_w
                                valid_w = (t_w >= 0) & (iw0 < W_in) & (iw0 * stride_w == t_w)

                                valid = mask_p & valid_d & valid_h & valid_w
                                x_mask = ci_mask[:, None] & valid[None, :]

                                x_ptrs = (
                                    x_base
                                    + offs_ci[:, None] * stride_xc
                                    + id0[None, :] * stride_xd
                                    + ih0[None, :] * stride_xh
                                    + iw0[None, :] * stride_xw
                                )
                                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                                # acc += sum_over_ci( w[ci,co,kd,kh,kw] * x[n,ci,id0,ih0,iw0] )
                                acc += tl.dot(w_t, x_vals, allow_tf32=True)

    # At this point, acc = sum over the 2x2x2 pooling window of conv_transpose(x; W)
    # (without conv_bias contribution)

    # Apply average pooling factor (1/8) and scale1
    factor = scale1 * (1.0 / 8.0)
    acc = acc * factor

    # Add conv_bias and post-pool bias in the correct order, then scale2:
    #   y_full = conv(x;W) + conv_bias
    #   y_scaled = y_full * scale1
    #   pooled = avg_pool(y_scaled)
    #   out = (pooled + bias_after) * scale2
    #
    # Since conv_bias is constant over spatial positions, avg_pool just keeps it:
    #   pooled = avg_pool(conv(x;W)*scale1) + conv_bias*scale1
    # We already accounted for avg_pool(conv(x;W)*scale1) in 'acc', so now:
    #   out = (acc + conv_bias*scale1 + bias_after) * scale2
    conv_b = tl.load(conv_b_ptr + offs_co, mask=mask_co, other=0.0)
    bias_after = tl.load(bias_after_ptr + offs_co, mask=mask_co, other=0.0)

    acc += conv_b[:, None] * scale1
    acc += bias_after[:, None]
    acc = acc * scale2

    # Store final output
    y_ptrs = (
        y_base
        + offs_co[:, None] * stride_yc
        + dp[None, :] * stride_yd
        + hp[None, :] * stride_yh
        + wp[None, :] * stride_yw
    )
    store_mask = mask_co[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def conv_transpose3d_avgpool3d_fused_triton(x, weight, conv_bias, bias_after, scale1, scale2, stride, padding):
    """
    Fused Triton wrapper for:

      ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        -> * scale1
        -> AvgPool3d(kernel_size=2, stride=2)
        -> + bias_after
        -> * scale2

    This computes the final pooled output directly, without materializing the
    high-resolution ConvTranspose3d output.
    """
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda and bias_after.is_cuda, \
        "Inputs must be CUDA tensors"

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not conv_bias.is_contiguous():
        conv_bias = conv_bias.contiguous()
    if not bias_after.is_contiguous():
        bias_after = bias_after.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    Ci_w, C_out, KD, KH, KW = weight.shape
    assert Ci_w == C_in, "Weight in_channels must match input channels"
    assert conv_bias.numel() == C_out
    assert bias_after.numel() == C_out

    s = int(stride)
    p = int(padding)

    # ConvTranspose3d output spatial sizes (full-resolution)
    D_full = (D_in - 1) * s - 2 * p + KD
    H_full = (H_in - 1) * s - 2 * p + KH
    W_full = (W_in - 1) * s - 2 * p + KW

    # AvgPool3d with kernel_size=2, stride=2, padding=0
    K_pool = 2
    S_pool = 2
    D_out = (D_full - K_pool) // S_pool + 1
    H_out = (H_full - K_pool) // S_pool + 1
    W_out = (W_full - K_pool) // S_pool + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    scale1_val = float(scale1) if isinstance(scale1, (torch.Tensor, float, int)) else float(scale1)
    scale2_val = float(scale2) if isinstance(scale2, (torch.Tensor, float, int)) else float(scale2)

    grid = lambda META: (
        triton.cdiv(D_out * H_out * W_out, META["BLOCK_P"]),  # pooled spatial tiles
        triton.cdiv(C_out, META["BLOCK_CO"]),                 # channel tiles
        max(1, N),                                            # batch
    )

    conv_transpose3d_avgpool3d_fused_kernel[grid](
        x, weight, conv_bias, bias_after, y,
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        s, s, s,
        p, p, p,
        D_full, H_full, W_full,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        scale1_val, scale2_val,
        BLOCK_CO=32,
        BLOCK_P=64,
        BLOCK_CI=32,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:

      ConvTranspose3d -> scale1 -> AvgPool3d(k=2,s=2) -> bias add -> scale2

    Behavior matches the reference PyTorch Model:

        self.conv_transpose = nn.ConvTranspose3d(...)
        self.scale1
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias
        self.scale2
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Match ConvTranspose3d weight layout: (in_channels, out_channels, KD, KH, KW)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        # Bias inside ConvTranspose3d
        self.conv_bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))

        # Scale after ConvTranspose3d
        self.scale1 = nn.Parameter(torch.tensor(float(scale1), dtype=torch.float32))

        # Bias added after AvgPool3d (shape: (out_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

        # Final scale
        self.scale2 = nn.Parameter(torch.tensor(float(scale2), dtype=torch.float32))

    def forward(self, x):
        # bias_after is broadcastable over (N, C_out, D_out, H_out, W_out)
        bias_vec = self.bias.view(self.out_channels)

        y = conv_transpose3d_avgpool3d_fused_triton(
            x,
            self.weight,
            self.conv_bias,
            bias_vec,
            self.scale1,
            self.scale2,
            self.stride,
            self.padding,
        )
        return y
