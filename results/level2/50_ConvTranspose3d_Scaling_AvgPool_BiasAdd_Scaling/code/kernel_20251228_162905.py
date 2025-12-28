import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scale_kernel(
    x_ptr, w_ptr, b_conv_ptr, y_ptr,
    N, C_in, D_in, H_in, W_in,
    C_out, KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wci, stride_wco, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    scale1,
    BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_CI: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_co = offs_co < C_out
    P_out = D_out * H_out * W_out
    mask_p = offs_p < P_out

    hw_out = H_out * W_out
    od = offs_p // hw_out
    rem = offs_p - od * hw_out
    oh = rem // W_out
    ow = rem - oh * W_out

    x_base = x_ptr + pid_n * stride_xn
    y_base = y_ptr + pid_n * stride_yn

    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    ci_range = tl.arange(0, BLOCK_CI)

    for ci_start in range(0, C_in, BLOCK_CI):
        offs_ci = ci_start + ci_range
        ci_mask = offs_ci < C_in

        w_ci_co_mask = ci_mask[:, None] & mask_co[None, :]

        for kd in range(0, KD):
            t_d = od + pad_d - kd
            id0 = t_d // stride_d
            valid_d = (t_d >= 0) & (id0 < D_in) & (id0 * stride_d == t_d)

            for kh in range(0, KH):
                t_h = oh + pad_h - kh
                ih0 = t_h // stride_h
                valid_h = (t_h >= 0) & (ih0 < H_in) & (ih0 * stride_h == t_h)

                for kw in range(0, KW):
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

                    w_ptrs = (
                        w_ptr
                        + offs_ci[:, None] * stride_wci
                        + offs_co[None, :] * stride_wco
                        + kd * stride_wkd
                        + kh * stride_wkh
                        + kw * stride_wkw
                    )
                    w_vals = tl.load(w_ptrs, mask=w_ci_co_mask, other=0.0)

                    w_t = tl.trans(w_vals)
                    acc += tl.dot(w_t, x_vals, allow_tf32=True)

    bias_vals = tl.load(b_conv_ptr + offs_co, mask=mask_co, other=0.0)
    acc += bias_vals[:, None]
    acc = acc * scale1

    y_ptrs = (
        y_base
        + offs_co[:, None] * stride_yc
        + od[None, :] * stride_yd
        + oh[None, :] * stride_yh
        + ow[None, :] * stride_yw
    )
    store_mask = mask_co[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


@triton.jit
def avgpool3d_bias_scale_kernel(
    x_ptr, bias_ptr, y_ptr,
    N, C, D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    scale2,
    BLOCK_CO: tl.constexpr, BLOCK_P: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    mask_co = offs_co < C
    P_out = D_out * H_out * W_out
    mask_p = offs_p < P_out

    hw_out = H_out * W_out
    dp = offs_p // hw_out
    rem = offs_p - dp * hw_out
    hp = rem // W_out
    wp = rem - hp * W_out

    x_base = x_ptr + pid_n * stride_xn
    y_base = y_ptr + pid_n * stride_yn

    acc = tl.zeros((BLOCK_CO, BLOCK_P), dtype=tl.float32)

    for dz in range(0, 2):
        id0 = dp * 2 + dz
        for dy in range(0, 2):
            ih0 = hp * 2 + dy
            for dx in range(0, 2):
                iw0 = wp * 2 + dx

                mask_xyz = (
                    mask_p
                    & (id0 < D_in)
                    & (ih0 < H_in)
                    & (iw0 < W_in)
                )
                load_mask = mask_co[:, None] & mask_xyz[None, :]

                x_ptrs = (
                    x_base
                    + offs_co[:, None] * stride_xc
                    + id0[None, :] * stride_xd
                    + ih0[None, :] * stride_xh
                    + iw0[None, :] * stride_xw
                )
                vals = tl.load(x_ptrs, mask=load_mask, other=0.0)
                acc += vals

    factor = 1.0 / 8.0
    acc = acc * factor

    bias_vals = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)
    acc += bias_vals[:, None]
    acc = acc * scale2

    y_ptrs = (
        y_base
        + offs_co[:, None] * stride_yc
        + dp[None, :] * stride_yd
        + hp[None, :] * stride_yh
        + wp[None, :] * stride_yw
    )
    store_mask = mask_co[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def conv_transpose3d_scale_triton(x, weight, conv_bias, scale1, stride, padding):
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda, "Inputs must be CUDA tensors"

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    Ci_w, C_out, KD, KH, KW = weight.shape
    assert Ci_w == C_in, "Weight in_channels must match input channels"

    s = int(stride)
    p = int(padding)

    D_out = (D_in - 1) * s - 2 * p + KD
    H_out = (H_in - 1) * s - 2 * p + KH
    W_out = (W_in - 1) * s - 2 * p + KW

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    scale1_val = float(scale1) if isinstance(scale1, (torch.Tensor, float, int)) else float(scale1)

    grid = lambda META: (
        triton.cdiv(D_out * H_out * W_out, META["BLOCK_P"]),
        triton.cdiv(C_out, META["BLOCK_CO"]),
        max(1, N),
    )

    conv_transpose3d_scale_kernel[grid](
        x, weight, conv_bias, y,
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        s, s, s,
        p, p, p,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        scale1_val,
        BLOCK_CO=32,
        BLOCK_P=64,
        BLOCK_CI=32,
        num_warps=4,
        num_stages=2,
    )
    return y


def avgpool3d_bias_scale2_triton(x, bias, scale2):
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"

    if not x.is_contiguous():
        x = x.contiguous()

    N, C, D_in, H_in, W_in = x.shape

    K = 2
    S = 2
    D_out = (D_in - K) // S + 1
    H_out = (H_in - K) // S + 1
    W_out = (W_in - K) // S + 1

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    scale2_val = float(scale2) if isinstance(scale2, (torch.Tensor, float, int)) else float(scale2)

    grid = lambda META: (
        triton.cdiv(D_out * H_out * W_out, META["BLOCK_P"]),
        triton.cdiv(C, META["BLOCK_CO"]),
        max(1, N),
    )

    avgpool3d_bias_scale_kernel[grid](
        x, bias, y,
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        scale2_val,
        BLOCK_CO=32,
        BLOCK_P=64,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
      ConvTranspose3d -> scale1 -> AvgPool3d(k=2,s=2) -> bias add -> scale2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.zeros(out_channels))

        self.scale1 = nn.Parameter(torch.tensor(float(scale1), dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(float(scale2), dtype=torch.float32))

    def forward(self, x):
        y = conv_transpose3d_scale_triton(
            x,
            self.weight,
            self.conv_bias,
            self.scale1,
            self.stride,
            self.padding,
        )
        bias_vec = self.bias.view(self.out_channels)
        y = avgpool3d_bias_scale2_triton(
            y,
            bias_vec,
            self.scale2,
        )
        return y
