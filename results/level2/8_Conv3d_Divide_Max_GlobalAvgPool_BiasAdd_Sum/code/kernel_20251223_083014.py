import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_out,
    D_out, H_out, W_out, P,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    C_IN: tl.constexpr,
    K_D: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    3D convolution: input [N, C_IN, D_in, H_in, W_in],
    weight [C_out, C_IN, K_D, K_H, K_W],
    output [N, C_out, D_out, H_out, W_out].

    P = N * D_out * H_out * W_out, flattened output positions.
    Grid: (cdiv(P, BLOCK_M), cdiv(C_out, BLOCK_N))
    """
    pid_m = tl.program_id(0)  # output position blocks
    pid_n = tl.program_id(1)  # output channel blocks (C_out)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    DHW = D_out * H_out * W_out
    HW = H_out * W_out

    n_idx = offs_m // DHW
    rem = offs_m % DHW
    od_idx = rem // HW
    rem2 = rem % HW
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ic in range(0, C_IN):
        for kd in range(0, K_D):
            id_idx = od_idx + kd
            for kh in range(0, K_H):
                ih_idx = oh_idx + kh
                for kw in range(0, K_W):
                    iw_idx = ow_idx + kw

                    x_offsets = (
                        n_idx * stride_xn
                        + ic * stride_xc
                        + id_idx * stride_xd
                        + ih_idx * stride_xh
                        + iw_idx * stride_xw
                    )
                    x_vals = tl.load(
                        x_ptr + x_offsets,
                        mask=mask_m,
                        other=0.0,
                    )

                    w_offsets = (
                        offs_n * stride_wo
                        + ic * stride_wc
                        + kd * stride_wd
                        + kh * stride_wh
                        + kw * stride_ww
                    )
                    w_vals = tl.load(
                        w_ptr + w_offsets,
                        mask=mask_n,
                        other=0.0,
                    )

                    acc += x_vals[:, None] * w_vals[None, :]

    # add conv bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # store output
    y_offsets = (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(
        y_ptr + y_offsets,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def conv3d_triton(x, weight, bias_conv):
    """
    x: [N, C_in, D_in, H_in, W_in]
    weight: [C_out, C_in, K_D, K_H, K_W]
    bias_conv: [C_out]
    returns: [N, C_out, D_out, H_out, W_out]
    """
    assert x.is_cuda and weight.is_cuda and bias_conv.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias_conv = bias_conv.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    K_D, K_H, K_W = weight.shape[2:]

    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wo, stride_wc, stride_wd, stride_wh, stride_ww = weight.stride()
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw = y.stride()

    P = N * D_out * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv3d_kernel[grid](
        x, weight, bias_conv, y,
        N, C_out,
        D_out, H_out, W_out, P,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wd, stride_wh, stride_ww,
        stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
        C_IN=C_in,
        K_D=K_D, K_H=K_H, K_W=K_W,
        BLOCK_M=64, BLOCK_N=32,
    )
    return y


@triton.jit
def maxpool3d_global_avg_kernel(
    x_ptr, out_ptr,
    N, C, D_out, H_out, W_out,
    Dp, Hp, Wp,
    KPD, KPH, KPW,
    divisor,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc,
):
    """
    For each (n, c): apply MaxPool3d with kernel=(KPD,KPH,KPW), stride=kernel,
    on conv output x (already computed), then divide by `divisor`, then
    global average over pooled spatial dims. Result: out[n, c].
    """
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Everything scalar in this kernel; reduction in registers
    sum_val = tl.zeros((), dtype=tl.float32)

    for pd in range(0, Dp):
        base_d = pd * KPD
        for ph in range(0, Hp):
            base_h = ph * KPH
            for pw in range(0, Wp):
                base_w = pw * KPW

                max_val = tl.full((), -1e30, dtype=tl.float32)

                for kd in range(0, KPD):
                    id = base_d + kd
                    for kh in range(0, KPH):
                        ih = base_h + kh
                        for kw in range(0, KPW):
                            iw = base_w + kw
                            offset = (
                                n * stride_xn
                                + c * stride_xc
                                + id * stride_xd
                                + ih * stride_xh
                                + iw * stride_xw
                            )
                            v = tl.load(x_ptr + offset)
                            v = v / divisor
                            max_val = tl.maximum(max_val, v)

                sum_val += max_val

    norm = 1.0 / (Dp * Hp * Wp)
    mean_val = sum_val * norm

    out_offset = n * stride_on + c * stride_oc
    tl.store(out_ptr + out_offset, mean_val)


def maxpool3d_global_avg_div(x, divisor, pool_kernel):
    """
    x: conv output [N, C_out, D_out, H_out, W_out]
    Applies: y = max_pool3d(x / divisor) then global avg pool to (1,1,1).
    Returns y_mean: [N, C_out]
    """
    assert x.is_cuda
    x = x.contiguous()
    N, C, D_out, H_out, W_out = x.shape
    KPD, KPH, KPW = pool_kernel

    # Assuming stride == kernel size (as in nn.MaxPool3d(pool_size) default)
    Dp = (D_out - KPD) // KPD + 1
    Hp = (H_out - KPH) // KPH + 1
    Wp = (W_out - KPW) // KPW + 1

    out = torch.empty((N, C), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_on, stride_oc = out.stride()

    grid = (triton.cdiv(N * C, 1),)
    maxpool3d_global_avg_kernel[grid](
        x, out,
        N, C, D_out, H_out, W_out,
        Dp, Hp, Wp,
        KPD, KPH, KPW,
        float(divisor),
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_on, stride_oc,
    )
    return out


@triton.jit
def bias_add_sum_kernel(
    v_ptr, bias_ptr, out_ptr,
    N, C,
    stride_vn, stride_vc,
    BLOCK_C: tl.constexpr,
):
    """
    v: [N, C] (result of pooled conv)
    bias: [C] (extra bias term broadcast over batch)
    out[n] = sum_c (v[n,c] + bias[c])
    """
    pid = tl.program_id(0)
    n = pid

    sum_val = tl.zeros((), dtype=tl.float32)
    offs_c = tl.arange(0, BLOCK_C)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + offs_c
        mask = (n < N) & (c_idx < C)

        v_vals = tl.load(
            v_ptr + n * stride_vn + c_idx * stride_vc,
            mask=mask,
            other=0.0,
        )
        b_vals = tl.load(
            bias_ptr + c_idx,
            mask=c_idx < C,
            other=0.0,
        )
        vals = v_vals + b_vals
        sum_val += tl.sum(vals, axis=0)

    tl.store(out_ptr + n, sum_val, mask=n < N)


def bias_add_sum(v, bias):
    """
    v: [N, C_out]
    bias: [C_out, 1, 1, 1]
    returns: [N] (sum over channel after adding bias)
    """
    assert v.is_cuda and bias.is_cuda
    v = v.contiguous()
    bias_vec = bias.view(-1).contiguous()

    N, C = v.shape
    out = torch.empty((N,), device=v.device, dtype=v.dtype)

    stride_vn, stride_vc = v.stride()

    def grid(meta):
        return (triton.cdiv(N, 1),)

    bias_add_sum_kernel[grid](
        v, bias_vec, out,
        N, C,
        stride_vn, stride_vc,
        BLOCK_C=32,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model:

    - 3D convolution (with bias)
    - division by constant
    - MaxPool3d
    - global average pooling to (1,1,1)
    - add bias term
    - sum along channel dimension
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor,
                 pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()

        # Initialize conv weights/bias using PyTorch's Conv3d init
        conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias_conv = nn.Parameter(conv.bias.detach().clone())

        self.divisor = float(divisor)
        self.pool_kernel = pool_size

        # Extra bias after global average pooling
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Kept for API compatibility; current implementation assumes sum_dim == 1
        self.sum_dim = sum_dim

    def forward(self, x):
        # Expect x on CUDA
        y = conv3d_triton(x, self.weight, self.bias_conv)
        v = maxpool3d_global_avg_div(y, self.divisor, self.pool_kernel)
        out_vec = bias_add_sum(v, self.bias)
        # Final shape [N, 1, 1, 1] to match original model
        return out_vec.view(out_vec.shape[0], 1, 1, 1)
