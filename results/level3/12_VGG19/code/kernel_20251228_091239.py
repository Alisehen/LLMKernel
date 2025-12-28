import torch
import torch.nn as nn
import triton
import triton.language as tl


# ===========================
# Triton Kernels
# ===========================

@triton.jit
def conv3x3_winograd_bias_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, H, W, Cout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    TH, TW,
    BLOCK_CO: tl.constexpr,
):
    """
    F(2x2,3x3) Winograd convolution with padding=1, stride=1.
    Each program instance:
      - processes one 2x2 output tile (per (n, tile_h, tile_w))
      - computes BLOCK_CO output channels
    """
    pid_t = tl.program_id(0)   # tile id
    pid_co = tl.program_id(1)  # output channel block id

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < Cout

    # Decode tile id -> (n, th, tw)
    tiles_per_n = TH * TW
    n = pid_t // tiles_per_n
    rem = pid_t % tiles_per_n
    th = rem // TW
    tw = rem % TW

    # Spatial origin of this 2x2 output tile
    oh0 = th * 2
    ow0 = tw * 2

    # Precompute input coordinates for 4x4 patch (with padding offset -1)
    ih0 = oh0 - 1
    ih1 = oh0 + 0
    ih2 = oh0 + 1
    ih3 = oh0 + 2

    iw0 = ow0 - 1
    iw1 = ow0 + 0
    iw2 = ow0 + 1
    iw3 = ow0 + 2

    # Accumulator in Winograd (4x4) domain: 16 values per output channel
    m00 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m01 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m02 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m03 = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    m10 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m11 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m12 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m13 = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    m20 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m21 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m22 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m23 = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    m30 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m31 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m32 = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    m33 = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, Cin):
        base_x = x_ptr + n * stride_xn + ci * stride_xc

        # Load 4x4 input tile with implicit padding=1 (scalar loads, no tensor indexing)
        # Row 0
        mask00 = (ih0 >= 0) & (ih0 < H) & (iw0 >= 0) & (iw0 < W)
        ptr00 = base_x + ih0 * stride_xh + iw0 * stride_xw
        d00 = tl.load(ptr00, mask=mask00, other=0.0).to(tl.float32)

        mask01 = (ih0 >= 0) & (ih0 < H) & (iw1 >= 0) & (iw1 < W)
        ptr01 = base_x + ih0 * stride_xh + iw1 * stride_xw
        d01 = tl.load(ptr01, mask=mask01, other=0.0).to(tl.float32)

        mask02 = (ih0 >= 0) & (ih0 < H) & (iw2 >= 0) & (iw2 < W)
        ptr02 = base_x + ih0 * stride_xh + iw2 * stride_xw
        d02 = tl.load(ptr02, mask=mask02, other=0.0).to(tl.float32)

        mask03 = (ih0 >= 0) & (ih0 < H) & (iw3 >= 0) & (iw3 < W)
        ptr03 = base_x + ih0 * stride_xh + iw3 * stride_xw
        d03 = tl.load(ptr03, mask=mask03, other=0.0).to(tl.float32)

        # Row 1
        mask10 = (ih1 >= 0) & (ih1 < H) & (iw0 >= 0) & (iw0 < W)
        ptr10 = base_x + ih1 * stride_xh + iw0 * stride_xw
        d10 = tl.load(ptr10, mask=mask10, other=0.0).to(tl.float32)

        mask11 = (ih1 >= 0) & (ih1 < H) & (iw1 >= 0) & (iw1 < W)
        ptr11 = base_x + ih1 * stride_xh + iw1 * stride_xw
        d11 = tl.load(ptr11, mask=mask11, other=0.0).to(tl.float32)

        mask12 = (ih1 >= 0) & (ih1 < H) & (iw2 >= 0) & (iw2 < W)
        ptr12 = base_x + ih1 * stride_xh + iw2 * stride_xw
        d12 = tl.load(ptr12, mask=mask12, other=0.0).to(tl.float32)

        mask13 = (ih1 >= 0) & (ih1 < H) & (iw3 >= 0) & (iw3 < W)
        ptr13 = base_x + ih1 * stride_xh + iw3 * stride_xw
        d13 = tl.load(ptr13, mask=mask13, other=0.0).to(tl.float32)

        # Row 2
        mask20 = (ih2 >= 0) & (ih2 < H) & (iw0 >= 0) & (iw0 < W)
        ptr20 = base_x + ih2 * stride_xh + iw0 * stride_xw
        d20 = tl.load(ptr20, mask=mask20, other=0.0).to(tl.float32)

        mask21 = (ih2 >= 0) & (ih2 < H) & (iw1 >= 0) & (iw1 < W)
        ptr21 = base_x + ih2 * stride_xh + iw1 * stride_xw
        d21 = tl.load(ptr21, mask=mask21, other=0.0).to(tl.float32)

        mask22 = (ih2 >= 0) & (ih2 < H) & (iw2 >= 0) & (iw2 < W)
        ptr22 = base_x + ih2 * stride_xh + iw2 * stride_xw
        d22 = tl.load(ptr22, mask=mask22, other=0.0).to(tl.float32)

        mask23 = (ih2 >= 0) & (ih2 < H) & (iw3 >= 0) & (iw3 < W)
        ptr23 = base_x + ih2 * stride_xh + iw3 * stride_xw
        d23 = tl.load(ptr23, mask=mask23, other=0.0).to(tl.float32)

        # Row 3
        mask30 = (ih3 >= 0) & (ih3 < H) & (iw0 >= 0) & (iw0 < W)
        ptr30 = base_x + ih3 * stride_xh + iw0 * stride_xw
        d30 = tl.load(ptr30, mask=mask30, other=0.0).to(tl.float32)

        mask31 = (ih3 >= 0) & (ih3 < H) & (iw1 >= 0) & (iw1 < W)
        ptr31 = base_x + ih3 * stride_xh + iw1 * stride_xw
        d31 = tl.load(ptr31, mask=mask31, other=0.0).to(tl.float32)

        mask32 = (ih3 >= 0) & (ih3 < H) & (iw2 >= 0) & (iw2 < W)
        ptr32 = base_x + ih3 * stride_xh + iw2 * stride_xw
        d32 = tl.load(ptr32, mask=mask32, other=0.0).to(tl.float32)

        mask33 = (ih3 >= 0) & (ih3 < H) & (iw3 >= 0) & (iw3 < W)
        ptr33 = base_x + ih3 * stride_xh + iw3 * stride_xw
        d33 = tl.load(ptr33, mask=mask33, other=0.0).to(tl.float32)

        # Winograd input transform: V = B^T * d * B (F(2x2,3x3))
        # Row transform: T = B^T * d
        # For each column j: [d0j, d1j, d2j, d3j] -> [d0j-d2j, d1j+d2j, d2j-d1j, d1j-d3j]
        # Row 0
        t0_0 = d00 - d20
        t0_1 = d01 - d21
        t0_2 = d02 - d22
        t0_3 = d03 - d23
        # Row 1
        t1_0 = d10 + d20
        t1_1 = d11 + d21
        t1_2 = d12 + d22
        t1_3 = d13 + d23
        # Row 2
        t2_0 = d20 - d10
        t2_1 = d21 - d11
        t2_2 = d22 - d12
        t2_3 = d23 - d13
        # Row 3
        t3_0 = d10 - d30
        t3_1 = d11 - d31
        t3_2 = d12 - d32
        t3_3 = d13 - d33

        # Column transform: V = T * B
        # For each row r: [c0,c1,c2,c3] -> [c0-c2, c1+c2, c2-c1, c1-c3]
        # Row 0
        v00 = t0_0 - t0_2
        v01 = t0_1 + t0_2
        v02 = t0_2 - t0_1
        v03 = t0_1 - t0_3
        # Row 1
        v10 = t1_0 - t1_2
        v11 = t1_1 + t1_2
        v12 = t1_2 - t1_1
        v13 = t1_1 - t1_3
        # Row 2
        v20 = t2_0 - t2_2
        v21 = t2_1 + t2_2
        v22 = t2_2 - t2_1
        v23 = t2_1 - t2_3
        # Row 3
        v30 = t3_0 - t3_2
        v31 = t3_1 + t3_2
        v32 = t3_2 - t3_1
        v33 = t3_1 - t3_3

        # Load corresponding transformed weights U(ci, co, 4,4)
        base_w_ci = w_ptr + ci * stride_wc

        # row 0
        w00_ptr = base_w_ci + 0 * stride_wh + 0 * stride_ww + offs_co * stride_wn
        u00 = tl.load(w00_ptr, mask=mask_co, other=0.0)
        m00 += u00 * v00

        w01_ptr = base_w_ci + 0 * stride_wh + 1 * stride_ww + offs_co * stride_wn
        u01 = tl.load(w01_ptr, mask=mask_co, other=0.0)
        m01 += u01 * v01

        w02_ptr = base_w_ci + 0 * stride_wh + 2 * stride_ww + offs_co * stride_wn
        u02 = tl.load(w02_ptr, mask=mask_co, other=0.0)
        m02 += u02 * v02

        w03_ptr = base_w_ci + 0 * stride_wh + 3 * stride_ww + offs_co * stride_wn
        u03 = tl.load(w03_ptr, mask=mask_co, other=0.0)
        m03 += u03 * v03

        # row 1
        w10_ptr = base_w_ci + 1 * stride_wh + 0 * stride_ww + offs_co * stride_wn
        u10 = tl.load(w10_ptr, mask=mask_co, other=0.0)
        m10 += u10 * v10

        w11_ptr = base_w_ci + 1 * stride_wh + 1 * stride_ww + offs_co * stride_wn
        u11 = tl.load(w11_ptr, mask=mask_co, other=0.0)
        m11 += u11 * v11

        w12_ptr = base_w_ci + 1 * stride_wh + 2 * stride_ww + offs_co * stride_wn
        u12 = tl.load(w12_ptr, mask=mask_co, other=0.0)
        m12 += u12 * v12

        w13_ptr = base_w_ci + 1 * stride_wh + 3 * stride_ww + offs_co * stride_wn
        u13 = tl.load(w13_ptr, mask=mask_co, other=0.0)
        m13 += u13 * v13

        # row 2
        w20_ptr = base_w_ci + 2 * stride_wh + 0 * stride_ww + offs_co * stride_wn
        u20 = tl.load(w20_ptr, mask=mask_co, other=0.0)
        m20 += u20 * v20

        w21_ptr = base_w_ci + 2 * stride_wh + 1 * stride_ww + offs_co * stride_wn
        u21 = tl.load(w21_ptr, mask=mask_co, other=0.0)
        m21 += u21 * v21

        w22_ptr = base_w_ci + 2 * stride_wh + 2 * stride_ww + offs_co * stride_wn
        u22 = tl.load(w22_ptr, mask=mask_co, other=0.0)
        m22 += u22 * v22

        w23_ptr = base_w_ci + 2 * stride_wh + 3 * stride_ww + offs_co * stride_wn
        u23 = tl.load(w23_ptr, mask=mask_co, other=0.0)
        m23 += u23 * v23

        # row 3
        w30_ptr = base_w_ci + 3 * stride_wh + 0 * stride_ww + offs_co * stride_wn
        u30 = tl.load(w30_ptr, mask=mask_co, other=0.0)
        m30 += u30 * v30

        w31_ptr = base_w_ci + 3 * stride_wh + 1 * stride_ww + offs_co * stride_wn
        u31 = tl.load(w31_ptr, mask=mask_co, other=0.0)
        m31 += u31 * v31

        w32_ptr = base_w_ci + 3 * stride_wh + 2 * stride_ww + offs_co * stride_wn
        u32 = tl.load(w32_ptr, mask=mask_co, other=0.0)
        m32 += u32 * v32

        w33_ptr = base_w_ci + 3 * stride_wh + 3 * stride_ww + offs_co * stride_wn
        u33 = tl.load(w33_ptr, mask=mask_co, other=0.0)
        m33 += u33 * v33

    # Inverse Winograd transform: Y = A^T * M * A
    # A^T = [[1, 1, 1, 0],
    #        [0, 1,-1,-1]]

    # Row transform: T = A^T * M  -> 2x4
    # First output row (t0*)
    t0c0 = m00 + m10 + m20
    t0c1 = m01 + m11 + m21
    t0c2 = m02 + m12 + m22
    t0c3 = m03 + m13 + m23

    # Second output row (t1*)
    t1c0 = m10 - m20 - m30
    t1c1 = m11 - m21 - m31
    t1c2 = m12 - m22 - m32
    t1c3 = m13 - m23 - m33

    # Column transform: Y = T * A -> 2x2
    y00 = t0c0 + t0c1 + t0c2
    y01 = t0c1 - t0c2 - t0c3
    y10 = t1c0 + t1c1 + t1c2
    y11 = t1c1 - t1c2 - t1c3

    # Add bias and apply ReLU
    bias = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)
    y00 = tl.maximum(y00 + bias, 0.0)
    y01 = tl.maximum(y01 + bias, 0.0)
    y10 = tl.maximum(y10 + bias, 0.0)
    y11 = tl.maximum(y11 + bias, 0.0)

    # Store results with boundary checks
    h0 = oh0
    h1 = oh0 + 1
    w0 = ow0
    w1 = ow0 + 1

    base_y_n = y_ptr + n * stride_yn

    mask_h0 = h0 < H
    mask_h1 = h1 < H
    mask_w0 = w0 < W
    mask_w1 = w1 < W

    # (h0, w0)
    y00_ptr = base_y_n + offs_co * stride_yc + h0 * stride_yh + w0 * stride_yw
    mask_y00 = mask_co & mask_h0 & mask_w0
    tl.store(y00_ptr, y00, mask=mask_y00)

    # (h0, w1)
    y01_ptr = base_y_n + offs_co * stride_yc + h0 * stride_yh + w1 * stride_yw
    mask_y01 = mask_co & mask_h0 & mask_w1
    tl.store(y01_ptr, y01, mask=mask_y01)

    # (h1, w0)
    y10_ptr = base_y_n + offs_co * stride_yc + h1 * stride_yh + w0 * stride_yw
    mask_y10 = mask_co & mask_h1 & mask_w0
    tl.store(y10_ptr, y10, mask=mask_y10)

    # (h1, w1)
    y11_ptr = base_y_n + offs_co * stride_yc + h1 * stride_yh + w1 * stride_yw
    mask_y11 = mask_co & mask_h1 & mask_w1
    tl.store(y11_ptr, y11, mask=mask_y11)


@triton.jit
def maxpool2x2_kernel(
    x_ptr, y_ptr,
    M, N, C, H_in, W_in, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < M  # M = N * C * H_out * W_out

    hw_out = H_out * W_out

    nc = offs // hw_out
    rem = offs % hw_out
    oh = rem // W_out
    ow = rem % W_out

    n = nc // C
    c = nc % C

    ih0 = oh * 2
    iw0 = ow * 2

    base = (
        x_ptr
        + n * stride_xn
        + c * stride_xc
        + ih0 * stride_xh
        + iw0 * stride_xw
    )

    ptr0 = base
    ptr1 = base + stride_xw
    ptr2 = base + stride_xh
    ptr3 = base + stride_xh + stride_xw

    neg_inf = -1.0e30
    v0 = tl.load(ptr0, mask=mask, other=neg_inf)
    v1 = tl.load(ptr1, mask=mask, other=neg_inf)
    v2 = tl.load(ptr2, mask=mask, other=neg_inf)
    v3 = tl.load(ptr3, mask=mask, other=neg_inf)

    max1 = tl.maximum(v0, v1)
    max2 = tl.maximum(v2, v3)
    out = tl.maximum(max1, max2)

    y_ptrs = (
        y_ptr
        + n * stride_yn
        + c * stride_yc
        + oh * stride_yh
        + ow * stride_yw
    )
    tl.store(y_ptrs, out, mask=mask)


@triton.jit
def fused_gemm_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kmask = offs_k[None, :] + k < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & kmask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=kmask.T & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def fused_gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        kmask = offs_k[None, :] + k < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & kmask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=kmask.T & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_c)


# ===========================
# Python Helpers / Wrappers
# ===========================

def _winograd_filter_transform(weight: torch.Tensor) -> torch.Tensor:
    """
    Transform 3x3 filters to Winograd F(2x2,3x3) domain:
        U = G * g * G^T
    weight: (Cout, Cin, 3, 3)
    returns: (Cout, Cin, 4, 4)
    """
    assert weight.ndim == 4 and weight.shape[2:] == (3, 3)
    Cout, Cin, _, _ = weight.shape

    device = weight.device
    # Use float32 for transform for numerical stability
    w32 = weight.to(torch.float32)

    G = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )  # (4,3)

    g = w32.view(-1, 3, 3)  # (Cout*Cin, 3,3)
    # temp = G * g
    temp = torch.einsum("ab,nbc->nac", G, g)          # (N,4,3)
    # U = temp * G^T
    U = torch.einsum("nab,bc->nac", temp, G.t())      # (N,4,4)

    U = U.view(Cout, Cin, 4, 4).contiguous()
    # Keep in float32; kernel also operates in float32
    return U


def conv3x3_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Winograd-optimized conv3x3 + bias + ReLU.
    x: (N, Cin, H, W), contiguous NCHW, float32
    weight: (Cout, Cin, 3, 3)
    bias: (Cout,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    N, Cin, H, W = x.shape
    Cout = weight.shape[0]

    # Pre-transform filters once per call
    w_win = _winograd_filter_transform(weight)  # (Cout, Cin, 4, 4)

    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)

    TH = (H + 1) // 2
    TW = (W + 1) // 2
    T = max(1, N * TH * TW)

    grid = lambda META: (
        T,
        triton.cdiv(Cout, META["BLOCK_CO"]),
    )

    conv3x3_winograd_bias_relu_kernel[grid](
        x, w_win, bias, y,
        N, Cin, H, W, Cout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_win.stride(0), w_win.stride(1), w_win.stride(2), w_win.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        TH, TW,
        BLOCK_CO=32,
        num_warps=4,
    )

    return y


def maxpool2x2(x: torch.Tensor) -> torch.Tensor:
    """
    2x2 max-pool, stride 2, no padding.
    x: (N, C, H, W)
    """
    assert x.is_cuda
    N, C, H_in, W_in = x.shape
    H_out = H_in // 2
    W_out = W_in // 2
    M = max(1, N * C * H_out * W_out)

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE"]),)

    maxpool2x2_kernel[grid](
        x, y,
        M, N, C, H_in, W_in, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_SIZE=256,
    )
    return y


def linear_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)
    bias: (N,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    b = weight.t().contiguous()  # (K, N)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    fused_gemm_bias_relu_kernel[grid](
        x, b, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    return y


def linear_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)
    bias: (N,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    b = weight.t().contiguous()  # (K, N)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    fused_gemm_bias_kernel[grid](
        x, b, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    return y


# ===========================
# VGG19 with Triton Kernels
# ===========================

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Convolutional blocks (VGG19)
        # Block 1
        self.conv1_1_weight = nn.Parameter(torch.randn(64, 3, 3, 3))
        self.conv1_1_bias = nn.Parameter(torch.zeros(64))
        self.conv1_2_weight = nn.Parameter(torch.randn(64, 64, 3, 3))
        self.conv1_2_bias = nn.Parameter(torch.zeros(64))

        # Block 2
        self.conv2_1_weight = nn.Parameter(torch.randn(128, 64, 3, 3))
        self.conv2_1_bias = nn.Parameter(torch.zeros(128))
        self.conv2_2_weight = nn.Parameter(torch.randn(128, 128, 3, 3))
        self.conv2_2_bias = nn.Parameter(torch.zeros(128))

        # Block 3
        self.conv3_1_weight = nn.Parameter(torch.randn(256, 128, 3, 3))
        self.conv3_1_bias = nn.Parameter(torch.zeros(256))
        self.conv3_2_weight = nn.Parameter(torch.randn(256, 256, 3, 3))
        self.conv3_2_bias = nn.Parameter(torch.zeros(256))
        self.conv3_3_weight = nn.Parameter(torch.randn(256, 256, 3, 3))
        self.conv3_3_bias = nn.Parameter(torch.zeros(256))
        self.conv3_4_weight = nn.Parameter(torch.randn(256, 256, 3, 3))
        self.conv3_4_bias = nn.Parameter(torch.zeros(256))

        # Block 4
        self.conv4_1_weight = nn.Parameter(torch.randn(512, 256, 3, 3))
        self.conv4_1_bias = nn.Parameter(torch.zeros(512))
        self.conv4_2_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv4_2_bias = nn.Parameter(torch.zeros(512))
        self.conv4_3_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv4_3_bias = nn.Parameter(torch.zeros(512))
        self.conv4_4_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv4_4_bias = nn.Parameter(torch.zeros(512))

        # Block 5
        self.conv5_1_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv5_1_bias = nn.Parameter(torch.zeros(512))
        self.conv5_2_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv5_2_bias = nn.Parameter(torch.zeros(512))
        self.conv5_3_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv5_3_bias = nn.Parameter(torch.zeros(512))
        self.conv5_4_weight = nn.Parameter(torch.randn(512, 512, 3, 3))
        self.conv5_4_bias = nn.Parameter(torch.zeros(512))

        # Classifier
        self.fc1_weight = nn.Parameter(torch.randn(4096, 512 * 7 * 7))
        self.fc1_bias = nn.Parameter(torch.zeros(4096))
        self.fc2_weight = nn.Parameter(torch.randn(4096, 4096))
        self.fc2_bias = nn.Parameter(torch.zeros(4096))
        self.fc3_weight = nn.Parameter(torch.randn(num_classes, 4096))
        self.fc3_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        # Features (all convs fused with bias+ReLU, pools as Triton maxpool)
        # Block 1
        x = conv3x3_bias_relu(x, self.conv1_1_weight, self.conv1_1_bias)
        x = conv3x3_bias_relu(x, self.conv1_2_weight, self.conv1_2_bias)
        x = maxpool2x2(x)

        # Block 2
        x = conv3x3_bias_relu(x, self.conv2_1_weight, self.conv2_1_bias)
        x = conv3x3_bias_relu(x, self.conv2_2_weight, self.conv2_2_bias)
        x = maxpool2x2(x)

        # Block 3
        x = conv3x3_bias_relu(x, self.conv3_1_weight, self.conv3_1_bias)
        x = conv3x3_bias_relu(x, self.conv3_2_weight, self.conv3_2_bias)
        x = conv3x3_bias_relu(x, self.conv3_3_weight, self.conv3_3_bias)
        x = conv3x3_bias_relu(x, self.conv3_4_weight, self.conv3_4_bias)
        x = maxpool2x2(x)

        # Block 4
        x = conv3x3_bias_relu(x, self.conv4_1_weight, self.conv4_1_bias)
        x = conv3x3_bias_relu(x, self.conv4_2_weight, self.conv4_2_bias)
        x = conv3x3_bias_relu(x, self.conv4_3_weight, self.conv4_3_bias)
        x = conv3x3_bias_relu(x, self.conv4_4_weight, self.conv4_4_bias)
        x = maxpool2x2(x)

        # Block 5
        x = conv3x3_bias_relu(x, self.conv5_1_weight, self.conv5_1_bias)
        x = conv3x3_bias_relu(x, self.conv5_2_weight, self.conv5_2_bias)
        x = conv3x3_bias_relu(x, self.conv5_3_weight, self.conv5_3_bias)
        x = conv3x3_bias_relu(x, self.conv5_4_weight, self.conv5_4_bias)
        x = maxpool2x2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier (Linear + bias fused with ReLU where applicable)
        x = linear_bias_relu(x, self.fc1_weight, self.fc1_bias)
        x = linear_bias_relu(x, self.fc2_weight, self.fc2_bias)
        x = linear_bias(x, self.fc3_weight, self.fc3_bias)

        return x
