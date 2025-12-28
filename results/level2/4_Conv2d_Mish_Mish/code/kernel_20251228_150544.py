import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_mish2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, H, W,
    C_out, KH, KW,
    H_out, W_out,
    K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_out

    # Map linear output index -> (n, oh, ow)
    hw_out = H_out * W_out
    n_idx = offs_m // hw_out
    rem_m = offs_m % hw_out
    oh = rem_m // W_out
    ow = rem_m % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Map K index -> (ci, kh, kw)
        kk_hw = KH * KW
        ci = offs_k // kk_hw
        rem_k = offs_k % kk_hw
        kh = rem_k // KW
        kw = rem_k % KW

        # Input pointers (B, C_in, H, W)
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ci[None, :] * stride_xc
            + (oh[:, None] + kh[None, :]) * stride_xh
            + (ow[:, None] + kw[None, :]) * stride_xw
        )

        # Weight pointers (C_out, C_in, KH, KW)
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ci[:, None] * stride_wc
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
        )

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x, w, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    # First Mish
    exp_acc = tl.exp(acc)
    sp = tl.log(1.0 + exp_acc)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    acc = acc * th

    # Second Mish
    exp_acc2 = tl.exp(acc)
    sp2 = tl.log(1.0 + exp_acc2)
    t22 = tl.exp(2.0 * sp2)
    th2 = (t22 - 1.0) / (t22 + 1.0)
    acc = acc * th2

    # Store result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )

    tl.store(
        y_ptrs,
        acc.to(tl.float32),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def conv2d_winograd_f2x2_3x3_mish2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, H, W,
    C_out, H_out, W_out,
    n_tiles_h, n_tiles_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    """
    Winograd F(2x2,3x3) convolution kernel for:
      - NCHW layout
      - stride=1, padding=0, groups=1
      - kernel_size == 3
      - H_out and W_out even (so 2x2 tiles cover the output)
    Computes conv2d + Mish + Mish in a single kernel.
    """

    pid_tile = tl.program_id(0)
    pid_co_block = tl.program_id(1)

    tile_per_batch = n_tiles_h * n_tiles_w

    # Decode tile -> (n, tile_h, tile_w)
    n = pid_tile // tile_per_batch
    tile_in_batch = pid_tile % tile_per_batch
    tile_h_idx = tile_in_batch // n_tiles_w
    tile_w_idx = tile_in_batch % n_tiles_w

    oh0 = tile_h_idx * 2
    ow0 = tile_w_idx * 2

    co_start = pid_co_block * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    co_mask = offs_co < C_out

    # Accumulator in Winograd domain: 4x4 per output channel block
    M0 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M1 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M2 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M3 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M4 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M5 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M6 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M7 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M8 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M9 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M10 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M11 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M12 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M13 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M14 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    M15 = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C_in):
        # ----- Input transform: V = B^T * d * B -----
        # Load a 4x4 patch for this tile and channel (scalar loads, no tensor indexing)
        base_in = x_ptr + n * stride_xn + ci * stride_xc

        h0 = oh0 + 0
        h1 = oh0 + 1
        h2 = oh0 + 2
        h3 = oh0 + 3
        w0 = ow0 + 0
        w1 = ow0 + 1
        w2 = ow0 + 2
        w3 = ow0 + 3

        d00 = tl.load(base_in + h0 * stride_xh + w0 * stride_xw)
        d01 = tl.load(base_in + h0 * stride_xh + w1 * stride_xw)
        d02 = tl.load(base_in + h0 * stride_xh + w2 * stride_xw)
        d03 = tl.load(base_in + h0 * stride_xh + w3 * stride_xw)

        d10 = tl.load(base_in + h1 * stride_xh + w0 * stride_xw)
        d11 = tl.load(base_in + h1 * stride_xh + w1 * stride_xw)
        d12 = tl.load(base_in + h1 * stride_xh + w2 * stride_xw)
        d13 = tl.load(base_in + h1 * stride_xh + w3 * stride_xw)

        d20 = tl.load(base_in + h2 * stride_xh + w0 * stride_xw)
        d21 = tl.load(base_in + h2 * stride_xh + w1 * stride_xw)
        d22 = tl.load(base_in + h2 * stride_xh + w2 * stride_xw)
        d23 = tl.load(base_in + h2 * stride_xh + w3 * stride_xw)

        d30 = tl.load(base_in + h3 * stride_xh + w0 * stride_xw)
        d31 = tl.load(base_in + h3 * stride_xh + w1 * stride_xw)
        d32 = tl.load(base_in + h3 * stride_xh + w2 * stride_xw)
        d33 = tl.load(base_in + h3 * stride_xh + w3 * stride_xw)

        # B^T * d (row transform)
        m00 = d00 - d20
        m01 = d01 - d21
        m02 = d02 - d22
        m03 = d03 - d23

        m10 = d10 + d20
        m11 = d11 + d21
        m12 = d12 + d22
        m13 = d13 + d23

        m20 = -d10 + d20
        m21 = -d11 + d21
        m22 = -d12 + d22
        m23 = -d13 + d23

        m30 = d10 - d30
        m31 = d11 - d31
        m32 = d12 - d32
        m33 = d13 - d33

        # (B^T*d)*B (column transform)
        V00 = m00 - m02
        V01 = m01 + m02
        V02 = -m01 + m02
        V03 = m01 - m03

        V10 = m10 - m12
        V11 = m11 + m12
        V12 = -m11 + m12
        V13 = m11 - m13

        V20 = m20 - m22
        V21 = m21 + m22
        V22 = -m21 + m22
        V23 = m21 - m23

        V30 = m30 - m32
        V31 = m31 + m32
        V32 = -m31 + m32
        V33 = m31 - m33

        # ----- Filter transform: U = G * g * G^T -----
        base_w = w_ptr + offs_co * stride_wo + ci * stride_wc

        g00 = tl.load(base_w + 0 * stride_wkh + 0 * stride_wkw, mask=co_mask, other=0.0)
        g01 = tl.load(base_w + 0 * stride_wkh + 1 * stride_wkw, mask=co_mask, other=0.0)
        g02 = tl.load(base_w + 0 * stride_wkh + 2 * stride_wkw, mask=co_mask, other=0.0)
        g10 = tl.load(base_w + 1 * stride_wkh + 0 * stride_wkw, mask=co_mask, other=0.0)
        g11 = tl.load(base_w + 1 * stride_wkh + 1 * stride_wkw, mask=co_mask, other=0.0)
        g12 = tl.load(base_w + 1 * stride_wkh + 2 * stride_wkw, mask=co_mask, other=0.0)
        g20 = tl.load(base_w + 2 * stride_wkh + 0 * stride_wkw, mask=co_mask, other=0.0)
        g21 = tl.load(base_w + 2 * stride_wkh + 1 * stride_wkw, mask=co_mask, other=0.0)
        g22 = tl.load(base_w + 2 * stride_wkh + 2 * stride_wkw, mask=co_mask, other=0.0)

        half = 0.5

        # t = G * g  (4x3)
        t00 = g00
        t01 = g01
        t02 = g02

        t10 = half * (g00 + g10 + g20)
        t11 = half * (g01 + g11 + g21)
        t12 = half * (g02 + g12 + g22)

        t20 = half * (g00 - g10 + g20)
        t21 = half * (g01 - g11 + g21)
        t22 = half * (g02 - g12 + g22)

        t30 = g20
        t31 = g21
        t32 = g22

        # U = t * G^T  (4x4)
        U00 = t00
        U01 = half * (t00 + t01 + t02)
        U02 = half * (t00 - t01 + t02)
        U03 = t02

        U10 = t10
        U11 = half * (t10 + t11 + t12)
        U12 = half * (t10 - t11 + t12)
        U13 = t12

        U20 = t20
        U21 = half * (t20 + t21 + t22)
        U22 = half * (t20 - t21 + t22)
        U23 = t22

        U30 = t30
        U31 = half * (t30 + t31 + t32)
        U32 = half * (t30 - t31 + t32)
        U33 = t32

        # ----- Elementwise multiply & accumulate over C_in -----
        M0 += U00 * V00
        M1 += U01 * V01
        M2 += U02 * V02
        M3 += U03 * V03

        M4 += U10 * V10
        M5 += U11 * V11
        M6 += U12 * V12
        M7 += U13 * V13

        M8 += U20 * V20
        M9 += U21 * V21
        M10 += U22 * V22
        M11 += U23 * V23

        M12 += U30 * V30
        M13 += U31 * V31
        M14 += U32 * V32
        M15 += U33 * V33

    # ----- Inverse Winograd transform: Y = A^T * M * A -----
    # A^T = [[1, 1, 1, 0],
    #        [0, 1, -1, -1]]

    # T = A^T * M  (4x2)
    T00 = M0 + M1 + M2
    T01 = M1 - M2 - M3

    T10 = M4 + M5 + M6
    T11 = M5 - M6 - M7

    T20 = M8 + M9 + M10
    T21 = M9 - M10 - M11

    T30 = M12 + M13 + M14
    T31 = M13 - M14 - M15

    # Y = T * A  (2x2)
    Y00 = T00 + T10 + T20
    Y01 = T01 + T11 + T21
    Y10 = T10 - T20 - T30
    Y11 = T11 - T21 - T31

    # Add bias
    bias = tl.load(b_ptr + offs_co, mask=co_mask, other=0.0)

    Y00 = Y00 + bias
    Y01 = Y01 + bias
    Y10 = Y10 + bias
    Y11 = Y11 + bias

    # First Mish
    exp_y00 = tl.exp(Y00)
    sp = tl.log(1.0 + exp_y00)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y00 = Y00 * th

    exp_y01 = tl.exp(Y01)
    sp = tl.log(1.0 + exp_y01)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y01 = Y01 * th

    exp_y10 = tl.exp(Y10)
    sp = tl.log(1.0 + exp_y10)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y10 = Y10 * th

    exp_y11 = tl.exp(Y11)
    sp = tl.log(1.0 + exp_y11)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y11 = Y11 * th

    # Second Mish
    exp_y00 = tl.exp(Y00)
    sp = tl.log(1.0 + exp_y00)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y00 = Y00 * th

    exp_y01 = tl.exp(Y01)
    sp = tl.log(1.0 + exp_y01)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y01 = Y01 * th

    exp_y10 = tl.exp(Y10)
    sp = tl.log(1.0 + exp_y10)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y10 = Y10 * th

    exp_y11 = tl.exp(Y11)
    sp = tl.log(1.0 + exp_y11)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    Y11 = Y11 * th

    # Store 2x2 output tile
    base_out = y_ptr + n * stride_yn + offs_co * stride_yc

    ptr00 = base_out + oh0 * stride_yh + ow0 * stride_yw
    ptr01 = base_out + oh0 * stride_yh + (ow0 + 1) * stride_yw
    ptr10 = base_out + (oh0 + 1) * stride_yh + ow0 * stride_yw
    ptr11 = base_out + (oh0 + 1) * stride_yh + (ow0 + 1) * stride_yw

    tl.store(ptr00, Y00, mask=co_mask)
    tl.store(ptr01, Y01, mask=co_mask)
    tl.store(ptr10, Y10, mask=co_mask)
    tl.store(ptr11, Y11, mask=co_mask)


def conv2d_mish2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    # Assumes NCHW, groups=1, stride=1, padding=0
    B, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels must match weight channels"

    H_out = H - KH + 1
    W_out = W - KW + 1

    x_ = x.contiguous()
    w_ = weight.contiguous()
    b_ = bias.contiguous()

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Winograd F(2x2,3x3) fast path: kernel 3x3, stride=1, padding=0, even output sizes
    use_winograd = (
        KH == 3
        and KW == 3
        and H_out > 0
        and W_out > 0
        and (H_out % 2 == 0)
        and (W_out % 2 == 0)
    )

    if use_winograd:
        n_tiles_h = H_out // 2
        n_tiles_w = W_out // 2
        total_tiles = B * n_tiles_h * n_tiles_w

        BLOCK_CO = 32

        def grid(meta):
            return (
                max(1, total_tiles),
                triton.cdiv(C_out, meta["BLOCK_CO"]),
            )

        conv2d_winograd_f2x2_3x3_mish2_kernel[grid](
            x_, w_, b_, y,
            B, C_in, H, W,
            C_out, H_out, W_out,
            n_tiles_h, n_tiles_w,
            x_.stride(0), x_.stride(1), x_.stride(2), x_.stride(3),
            w_.stride(0), w_.stride(1), w_.stride(2), w_.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_CO=BLOCK_CO,
        )
    else:
        # Generic direct-convolution fallback
        K = C_in * KH * KW

        def grid(meta):
            return (
                triton.cdiv(B * H_out * W_out, meta["BLOCK_M"]),
                triton.cdiv(C_out, meta["BLOCK_N"]),
            )

        conv2d_mish2_kernel[grid](
            x_, w_, b_, y,
            B, C_in, H, W,
            C_out, KH, KW,
            H_out, W_out,
            K,
            x_.stride(0), x_.stride(1), x_.stride(2), x_.stride(3),
            w_.stride(0), w_.stride(1), w_.stride(2), w_.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    Conv2d (stride=1, padding=0, groups=1) + Mish + Mish fused in a single kernel.
    Uses Winograd F(2x2,3x3) for 3x3 convolutions when applicable.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return conv2d_mish2(x, self.weight, self.bias)
