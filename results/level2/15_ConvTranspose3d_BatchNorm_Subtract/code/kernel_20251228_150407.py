import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc,  # w: (K, C_out) with strides (stride_wn, stride_wc)
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a BLOCK_M x BLOCK_N tile of the GEMM:
    #  A: (M, K) implicit from x (im2col of transposed conv)
    #  B: (K, C_out) = flattened weights
    #  C: (M, C_out) = y reshaped as (N*D_out*H_out*W_out, C_out)
    pid_m = tl.program_id(0)  # tile index along output positions (M)
    pid_n = tl.program_id(1)  # tile index along output channels (C_out)

    M = N * D_out * H_out * W_out
    K_total = C_in * kD * kH * kW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode flattened output position index offs_m -> (n, z_out, y_out, x_out)
    tmp = offs_m
    x_out = tmp % W_out
    tmp = tmp // W_out
    y_out = tmp % H_out
    tmp = tmp // H_out
    z_out = tmp % D_out
    tmp = tmp // D_out
    n_idx = tmp  # [BLOCK_M]

    # Prepare 2D broadcasted versions for later use
    x_out_b = tl.reshape(x_out, (BLOCK_M, 1))
    y_out_b = tl.reshape(y_out, (BLOCK_M, 1))
    z_out_b = tl.reshape(z_out, (BLOCK_M, 1))
    n_idx_b = tl.reshape(n_idx, (BLOCK_M, 1))
    mask_m_2d = tl.reshape(mask_m, (BLOCK_M, 1))

    # Accumulator for C tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        # Decode flattened kernel index offs_k -> (c_in, kz, ky, kx)
        tmpk = offs_k
        kx = tmpk % kW
        tmpk = tmpk // kW
        ky = tmpk % kH
        tmpk = tmpk // kH
        kz = tmpk % kD
        tmpk = tmpk // kD
        c_in = tmpk  # [BLOCK_K]

        kx_b = tl.reshape(kx, (1, BLOCK_K))
        ky_b = tl.reshape(ky, (1, BLOCK_K))
        kz_b = tl.reshape(kz, (1, BLOCK_K))
        c_in_b = tl.reshape(c_in, (1, BLOCK_K))
        mask_k_2d = tl.reshape(mask_k, (1, BLOCK_K))

        # Compute corresponding input spatial indices for transposed conv
        dz = z_out_b + pad_d - kz_b
        dy = y_out_b + pad_h - ky_b
        dx = x_out_b + pad_w - kx_b

        z_in = dz // stride_d
        y_in = dy // stride_h
        x_in = dx // stride_w

        mask_z = (dz >= 0) & (z_in >= 0) & (z_in < D_in) & (z_in * stride_d == dz)
        mask_y = (dy >= 0) & (y_in >= 0) & (y_in < H_in) & (y_in * stride_h == dy)
        mask_x = (dx >= 0) & (x_in >= 0) & (x_in < W_in) & (x_in * stride_w == dx)

        mask_valid = mask_m_2d & mask_k_2d & mask_z & mask_y & mask_x

        # Compute input offsets: x[n, c_in, z_in, y_in, x_in]
        offs_x = (
            n_idx_b * stride_xn
            + c_in_b * stride_xc
            + z_in * stride_xd
            + y_in * stride_xh
            + x_in * stride_xw
        )
        a = tl.load(x_ptr + offs_x, mask=mask_valid, other=0.0)

        # Load weight tile B: w[offs_k, offs_n]
        offs_w = offs_k[:, None] * stride_wn + offs_n[None, :] * stride_wc
        mask_w = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + offs_w, mask=mask_w, other=0.0)

        # GEMM update: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N)
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32), allow_tf32=True)

    # Add bias if present: bias[c_out]
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        bias_b = tl.reshape(bias, (1, BLOCK_N))
        acc += bias_b

    # Compute output offsets and store: y[n, c_out, z_out, y_out, x_out]
    c_out_b = tl.reshape(offs_n, (1, BLOCK_N))
    offs_y = (
        n_idx_b * stride_yn
        + c_out_b * stride_yc
        + z_out_b * stride_yd
        + y_out_b * stride_yh
        + x_out_b * stride_yw
    )
    mask = mask_m_2d & (tl.reshape(mask_n, (1, BLOCK_N)))
    tl.store(y_ptr + offs_y, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    """
    x: (N, C_in, D_in, H_in, W_in)
    weight: (C_in, C_out, kD, kH, kW) - same layout as nn.ConvTranspose3d.weight
    bias: (C_out,) or None
    stride, padding: int or 3-tuple
    """
    assert x.is_cuda and weight.is_cuda
    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, kD, kH, kW = weight.shape
    assert C_in_w == C_in

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding

    # PyTorch ConvTranspose3d output size (dilation=1, output_padding=0)
    D_out = (D_in - 1) * stride_d - 2 * pad_d + kD
    H_out = (H_in - 1) * stride_h - 2 * pad_h + kH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + kW

    # Output tensor
    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten weights to (K, C_out) for GEMM, where K = C_in * kD * kH * kW
    K_total = C_in * kD * kH * kW
    w_flat = weight.reshape(K_total, C_out).contiguous()

    x_strides = x.stride()
    w_strides = w_flat.stride()
    y_strides = y.stride()

    M = N * D_out * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    HAS_BIAS = bias is not None
    b_ptr = bias if HAS_BIAS else x  # dummy pointer when no bias

    conv_transpose3d_gemm_kernel[grid](
        x,
        w_flat,
        b_ptr,
        y,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        kD,
        kH,
        kW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        x_strides[0],
        x_strides[1],
        x_strides[2],
        x_strides[3],
        x_strides[4],
        w_strides[0],
        w_strides[1],
        y_strides[0],
        y_strides[1],
        y_strides[2],
        y_strides[3],
        y_strides[4],
        HAS_BIAS=HAS_BIAS,
        BLOCK_M=32,
        BLOCK_N=32,
        BLOCK_K=64,
        num_warps=4,
    )
    return y


@triton.jit
def spatial_mean_kernel(
    x_ptr, mean_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_mn, stride_mc,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = N * C
    mask_nc = pid < nc

    n = pid // C
    c = pid % C

    num_spatial = D * H * W

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for offset in range(0, num_spatial, BLOCK):
        idx = offset + tl.arange(0, BLOCK)
        mask = (idx < num_spatial) & mask_nc

        w_idx = idx % W
        tmp = idx // W
        h_idx = tmp % H
        d_idx = tmp // H

        x_offset = (
            n * stride_xn
            + c * stride_xc
            + d_idx * stride_xd
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        acc += x_val.to(tl.float32)

    total = tl.sum(acc, axis=0)
    denom = num_spatial
    mean = total / denom

    m_offset = n * stride_mn + c * stride_mc
    tl.store(mean_ptr + m_offset, mean, mask=mask_nc)


@triton.jit
def subtract_mean_kernel(
    x_ptr, mean_ptr, y_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_mn, stride_mc,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    numel = N * C * D * H * W
    mask = offs < numel

    w_idx = offs % W
    tmp = offs // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    tmp = tmp // D
    c_idx = tmp % C
    n_idx = tmp // C

    x_offset = (
        n_idx * stride_xn
        + c_idx * stride_xc
        + d_idx * stride_xd
        + h_idx * stride_xh
        + w_idx * stride_xw
    )
    m_offset = n_idx * stride_mn + c_idx * stride_mc
    y_offset = (
        n_idx * stride_yn
        + c_idx * stride_yc
        + d_idx * stride_yd
        + h_idx * stride_yh
        + w_idx * stride_yw
    )

    x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + m_offset, mask=mask, other=0.0)
    out = x_val - mean
    tl.store(y_ptr + y_offset, out, mask=mask)


def subtract_spatial_mean_triton(x):
    """
    x: (N, C, D, H, W)
    Returns x - mean(x, dim=(2,3,4), keepdim=True)
    """
    assert x.is_cuda
    N, C, D, H, W = x.shape
    mean = torch.empty((N, C), device=x.device, dtype=x.dtype)

    x_strides = x.stride()
    m_strides = mean.stride()

    grid_mean = lambda META: (max(1, N * C),)
    spatial_mean_kernel[grid_mean](
        x,
        mean,
        N,
        C,
        D,
        H,
        W,
        x_strides[0],
        x_strides[1],
        x_strides[2],
        x_strides[3],
        x_strides[4],
        m_strides[0],
        m_strides[1],
        BLOCK=256,
        num_warps=2,
    )

    y = torch.empty_like(x)
    y_strides = y.stride()
    numel = N * C * D * H * W
    grid_sub = lambda META: (triton.cdiv(numel, META["BLOCK"]),)
    subtract_mean_kernel[grid_sub](
        x,
        mean,
        y,
        N,
        C,
        D,
        H,
        W,
        x_strides[0],
        x_strides[1],
        x_strides[2],
        x_strides[3],
        x_strides[4],
        m_strides[0],
        m_strides[1],
        y_strides[0],
        y_strides[1],
        y_strides[2],
        y_strides[3],
        y_strides[4],
        BLOCK=256,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    """
    ConvTranspose3d implemented with an optimized Triton GEMM-style kernel,
    followed by PyTorch BatchNorm3d, then Triton-based spatial mean subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # ConvTranspose3d via optimized Triton kernel
        y = conv_transpose3d_triton(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
        )
        # BatchNorm3d in PyTorch for correctness and running stats
        y = self.batch_norm(y)
        # Subtract spatial mean via Triton
        y = subtract_spatial_mean_triton(y)
        return y
