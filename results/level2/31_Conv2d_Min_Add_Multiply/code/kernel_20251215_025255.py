import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small, very register-safe tile (fallback)
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=1,
        ),
        # Wider in output channels
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=1,
        ),
        # Taller in spatial positions
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=['C_in', 'H_out', 'W_out', 'C_out'],
)
@triton.jit
def conv2d_min_bias_scale_kernel(
    x_ptr,            # [N, C_in, H_in, W_in]
    w_ptr,            # [C_out, C_in, K_H, K_W]
    conv_bias_ptr,    # [C_out]
    extra_bias_ptr,   # [C_out, 1, 1]
    out_ptr,          # [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_cb0,
    stride_eb0,
    stride_on, stride_oc, stride_oh, stride_ow,
    const_val,        # scalar
    scaling,          # scalar
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
):
    # program ids:
    #  - pid_m tiles over flattened spatial positions P = N * H_out * W_out
    #  - pid_n tiles over output channels C_out
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * H_out * W_out
    valid_m = offs_m < P
    valid_n = offs_n < C_out

    # decode offs_m -> (n_idx, oh_idx, ow_idx)
    DHW = H_out * W_out
    n_idx = offs_m // DHW
    rem = offs_m % DHW
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # main convolution loop: sum over input channels and kernel spatial dims
    # K_H and K_W are constexpr -> inner loops are unrolled
    for ic in range(0, C_in):
        for kh in tl.static_range(0, K_H):
            ih = oh_idx + kh
            for kw in tl.static_range(0, K_W):
                iw = ow_idx + kw

                # pointers to input x: [BLOCK_M, 1]
                x_ptrs = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + ic * stride_xc
                    + ih[:, None] * stride_xh
                    + iw[:, None] * stride_xw
                )
                x_vals = tl.load(
                    x_ptrs,
                    mask=valid_m[:, None],
                    other=0.0,
                ).to(tl.float32)

                # pointers to weight w: [1, BLOCK_N]
                w_ptrs = (
                    w_ptr
                    + offs_n[None, :] * stride_wo
                    + ic * stride_wc
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_vals = tl.load(
                    w_ptrs,
                    mask=valid_n[None, :],
                    other=0.0,
                ).to(tl.float32)

                # outer-product accumulate
                acc += x_vals * w_vals

    # add convolution bias (before min)
    cb = tl.load(
        conv_bias_ptr + offs_n * stride_cb0,
        mask=valid_n,
        other=0.0,
    ).to(tl.float32)
    acc += cb[None, :]

    # apply min with constant value (elementwise)
    acc = tl.where(acc < const_val, acc, const_val)

    # add extra bias (after min)
    eb = tl.load(
        extra_bias_ptr + offs_n * stride_eb0,
        mask=valid_n,
        other=0.0,
    ).to(tl.float32)
    acc += eb[None, :]

    # scale
    acc = acc * scaling

    # store result
    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + oh_idx[:, None] * stride_oh
        + ow_idx[:, None] * stride_ow
    )
    store_mask = valid_m[:, None] & valid_n[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def conv2d_min_bias_scale_triton(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    """
    Fused:
        y = conv2d(x, weight, bias=conv_bias, stride=1, padding=0, dilation=1, groups=1)
        y = min(y, constant_value)
        y = y + extra_bias
        y = y * scaling_factor

    Shapes:
      x          : [N, C_in, H_in, W_in]
      weight     : [C_out, C_in, K_H, K_W]
      conv_bias  : [C_out]
      extra_bias : [C_out, 1, 1]
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel"
    assert x.ndim == 4 and weight.ndim == 4
    assert conv_bias.ndim == 1
    assert extra_bias.ndim == 3

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in_w == C_in, "Weight C_in mismatch with input"

    # stride=1, padding=0, dilation=1, groups=1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert H_out > 0 and W_out > 0

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # use existing strides (no need to force contiguity as we pass explicit strides)
    x_c = x
    w_c = weight
    cb_c = conv_bias
    eb_c = extra_bias

    # strides
    sx0, sx1, sx2, sx3 = x_c.stride()
    sw0, sw1, sw2, sw3 = w_c.stride()
    scb0 = cb_c.stride(0)
    seb0 = eb_c.stride(0)
    so0, so1, so2, so3 = out.stride()

    # total number of output positions
    P = N * H_out * W_out

    # grid function uses autotuned BLOCK_M/BLOCK_N
    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv2d_min_bias_scale_kernel[grid](
        x_ptr=x_c,
        w_ptr=w_c,
        conv_bias_ptr=cb_c,
        extra_bias_ptr=eb_c,
        out_ptr=out,
        N=N,
        C_in=C_in,
        H_in=H_in,
        W_in=W_in,
        C_out=C_out,
        H_out=H_out,
        W_out=W_out,
        stride_xn=sx0,
        stride_xc=sx1,
        stride_xh=sx2,
        stride_xw=sx3,
        stride_wo=sw0,
        stride_wc=sw1,
        stride_wkh=sw2,
        stride_wkw=sw3,
        stride_cb0=scb0,
        stride_eb0=seb0,
        stride_on=so0,
        stride_oc=so1,
        stride_oh=so2,
        stride_ow=so3,
        const_val=float(constant_value),
        scaling=float(scaling_factor),
        K_H=K_H,
        K_W=K_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:

        x = conv2d(x)
        x = min(x, constant_value)
        x = x + bias
        x = x * scaling_factor

    where conv2d has its own bias term.
    """

    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)

        # Conv2d-like parameters (weight + conv bias)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k_h, k_w)
        )
        self.conv_bias = nn.Parameter(
            torch.randn(out_channels)
        )

        # Extra bias added after min, broadcast over spatial
        self.bias = nn.Parameter(torch.randn(*bias_shape))

        # Scalars
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # x is assumed to be [N, C_in, H, W], on CUDA
        return conv2d_min_bias_scale_triton(
            x,
            self.weight,
            self.conv_bias,
            self.bias,
            self.constant_value,
            self.scaling_factor,
        )
