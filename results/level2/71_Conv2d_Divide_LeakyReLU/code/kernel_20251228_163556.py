import torch, torch.nn as nn, triton, triton.language as tl
import math


# ----------------------------
# Direct GEMM-style Conv kernel
# ----------------------------

@triton.jit
def conv2d_div_leakyrelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, H_in, W_in,
    H_out, W_out,
    Cin, Cout,
    KH, KW,
    M, N, K,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    divisor, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < N

    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    rem_m = offs_m % HW_out
    ho_idx = rem_m // W_out
    wo_idx = rem_m % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    KHW = KH * KW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + k_range  # [BLOCK_K]
        k_mask = offs_k < K

        ci = offs_k // KHW
        rem_k = offs_k % KHW
        kh = rem_k // KW
        kw = rem_k % KW

        b_b = b_idx[:, None]
        ho_b = ho_idx[:, None]
        wo_b = wo_idx[:, None]

        ci_b = ci[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        h_in = ho_b + kh_b
        w_in = wo_b + kw_b

        a_ptrs = (
            x_ptr
            + b_b * stride_x_n
            + ci_b * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )
        a_mask = (mask_m[:, None]) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        co_b = offs_n[None, :]
        ci_k = ci[:, None]
        kh_k = kh[:, None]
        kw_k = kw[:, None]

        w_ptrs = (
            w_ptr
            + co_b * stride_w_cout
            + ci_k * stride_w_cin
            + kh_k * stride_w_kh
            + kw_k * stride_w_kw
        )
        b_mask = (k_mask[:, None]) & (mask_n[None, :])
        w = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, w, allow_tf32=True)

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    acc = acc / divisor
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    out_b = b_idx[:, None]
    out_co = offs_n[None, :]
    out_ho = ho_idx[:, None]
    out_wo = wo_idx[:, None]

    y_ptrs = (
        y_ptr
        + out_b * stride_y_n
        + out_co * stride_y_c
        + out_ho * stride_y_h
        + out_wo * stride_y_w
    )
    out_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(y_ptrs, acc, mask=out_mask)


# ----------------------------
# Winograd weight pre-transform
# ----------------------------

def precompute_winograd_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Precompute Winograd F(2x2,3x3) weight transform:
        U = G * g * G^T
    weight: (Cout, Cin, 3, 3)
    returns: (Cout, Cin, 4, 4)
    """
    Cout, Cin, KH, KW = weight.shape
    assert KH == 3 and KW == 3

    # Canonical F(2,3) G matrix (Lavin et al.)
    G = torch.tensor(
        [
            [1.0 / 4.0, 0.0, 0.0],
            [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
            [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
            [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
        ],
        dtype=weight.dtype,
        device=weight.device,
    )  # (4,3)
    GT = G.t()  # (3,4)

    g = weight.contiguous().view(-1, 3, 3)  # (Cout*Cin, 3, 3)
    F = g.shape[0]

    G_expand = G.unsqueeze(0).expand(F, -1, -1)   # (F,4,3)
    GT_expand = GT.unsqueeze(0).expand(F, -1, -1) # (F,3,4)

    tmp = torch.bmm(G_expand, g)      # (F,4,3)
    U = torch.bmm(tmp, GT_expand)     # (F,4,4)

    U = U.view(Cout, Cin, 4, 4).contiguous()
    return U


# ----------------------------
# Winograd F(2x2,3x3) kernel
# ----------------------------

@triton.jit
def winograd_f23_div_leakyrelu_kernel(
    x_ptr, wino_w_ptr, b_ptr, y_ptr,
    B, H_in, W_in,
    H_out, W_out,
    Cin, Cout,
    H_tiles, W_tiles, total_tiles,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_h, stride_w_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    divisor, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # tiles
    pid_n = tl.program_id(1)  # output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # tile indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # Cout indices

    mask_m = offs_m < total_tiles
    mask_n = offs_n < Cout

    tiles_per_batch = H_tiles * W_tiles

    b_idx = offs_m // tiles_per_batch
    rem = offs_m % tiles_per_batch
    tile_h = rem // W_tiles
    tile_w = rem % W_tiles

    # top-left output coord (2x2 tile)
    oh0 = tile_h * 2
    ow0 = tile_w * 2

    # 4x4 transform-domain accumulators M (4x4 for each tile, channel)
    m00 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m01 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m02 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m03 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m12 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m13 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m20 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m21 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m22 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m23 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m30 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m31 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m33 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over Cin
    for ci in range(0, Cin):
        # Load 4x4 input patch per tile (for this input channel)
        col_offsets = tl.arange(0, 4)  # 0..3

        # Row 0
        h0 = oh0
        base0 = (
            x_ptr
            + b_idx * stride_x_n
            + ci * stride_x_c
            + h0 * stride_x_h
            + ow0 * stride_x_w
        )
        ptrs0 = base0[:, None] + col_offsets[None, :] * stride_x_w
        row0 = tl.load(ptrs0, mask=mask_m[:, None], other=0.0)
        d00 = row0[:, 0]
        d01 = row0[:, 1]
        d02 = row0[:, 2]
        d03 = row0[:, 3]

        # Row 1
        h1 = oh0 + 1
        base1 = (
            x_ptr
            + b_idx * stride_x_n
            + ci * stride_x_c
            + h1 * stride_x_h
            + ow0 * stride_x_w
        )
        ptrs1 = base1[:, None] + col_offsets[None, :] * stride_x_w
        row1 = tl.load(ptrs1, mask=mask_m[:, None], other=0.0)
        d10 = row1[:, 0]
        d11 = row1[:, 1]
        d12 = row1[:, 2]
        d13 = row1[:, 3]

        # Row 2
        h2 = oh0 + 2
        base2 = (
            x_ptr
            + b_idx * stride_x_n
            + ci * stride_x_c
            + h2 * stride_x_h
            + ow0 * stride_x_w
        )
        ptrs2 = base2[:, None] + col_offsets[None, :] * stride_x_w
        row2 = tl.load(ptrs2, mask=mask_m[:, None], other=0.0)
        d20 = row2[:, 0]
        d21 = row2[:, 1]
        d22 = row2[:, 2]
        d23 = row2[:, 3]

        # Row 3
        h3 = oh0 + 3
        base3 = (
            x_ptr
            + b_idx * stride_x_n
            + ci * stride_x_c
            + h3 * stride_x_h
            + ow0 * stride_x_w
        )
        ptrs3 = base3[:, None] + col_offsets[None, :] * stride_x_w
        row3 = tl.load(ptrs3, mask=mask_m[:, None], other=0.0)
        d30 = row3[:, 0]
        d31 = row3[:, 1]
        d32 = row3[:, 2]
        d33 = row3[:, 3]

        # Input transform V = B^T * d * B
        # B from Lavin: [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
        # First: tmp = B^T * d  (left multiply)
        # rows of B^T are columns of B
        # tmp0 = d0
        tmp00 = d00
        tmp01 = d01
        tmp02 = d02
        tmp03 = d03

        # tmp1 = d1 - d2 + d3
        tmp10 = d10 - d20 + d30
        tmp11 = d11 - d21 + d31
        tmp12 = d12 - d22 + d32
        tmp13 = d13 - d23 + d33

        # tmp2 = -d0 + d1 + d2
        tmp20 = -d00 + d10 + d20
        tmp21 = -d01 + d11 + d21
        tmp22 = -d02 + d12 + d22
        tmp23 = -d03 + d13 + d23

        # tmp3 = -d3
        tmp30 = -d30
        tmp31 = -d31
        tmp32 = -d32
        tmp33 = -d33

        # Then: V = tmp * B (right multiply)
        # row-wise: r @ B
        # v_row0
        v00 = tmp00
        v01 = tmp01 - tmp02 + tmp03
        v02 = -tmp00 + tmp01 + tmp02
        v03 = -tmp03

        # v_row1
        v10 = tmp10
        v11 = tmp11 - tmp12 + tmp13
        v12 = -tmp10 + tmp11 + tmp12
        v13 = -tmp13

        # v_row2
        v20 = tmp20
        v21 = tmp21 - tmp22 + tmp23
        v22 = -tmp20 + tmp21 + tmp22
        v23 = -tmp23

        # v_row3
        v30 = tmp30
        v31 = tmp31 - tmp32 + tmp33
        v32 = -tmp30 + tmp31 + tmp32
        v33 = -tmp33

        # Accumulate elementwise products with transformed weights U[co, ci, r, c]
        co = offs_n[None, :]  # (1, BLOCK_N)

        # Helper to load U[r,c] vector for all Cout in this block
        def load_u(rc_h, rc_w):
            ptr = (
                wino_w_ptr
                + co * stride_w_cout
                + ci * stride_w_cin
                + rc_h * stride_w_h
                + rc_w * stride_w_w
            )
            return tl.load(ptr, mask=mask_n[None, :], other=0.0)[0, :]

        u00 = load_u(0, 0)
        u01 = load_u(0, 1)
        u02 = load_u(0, 2)
        u03 = load_u(0, 3)

        u10 = load_u(1, 0)
        u11 = load_u(1, 1)
        u12 = load_u(1, 2)
        u13 = load_u(1, 3)

        u20 = load_u(2, 0)
        u21 = load_u(2, 1)
        u22 = load_u(2, 2)
        u23 = load_u(2, 3)

        u30 = load_u(3, 0)
        u31 = load_u(3, 1)
        u32 = load_u(3, 2)
        u33 = load_u(3, 3)

        m00 += v00[:, None] * u00[None, :]
        m01 += v01[:, None] * u01[None, :]
        m02 += v02[:, None] * u02[None, :]
        m03 += v03[:, None] * u03[None, :]

        m10 += v10[:, None] * u10[None, :]
        m11 += v11[:, None] * u11[None, :]
        m12 += v12[:, None] * u12[None, :]
        m13 += v13[:, None] * u13[None, :]

        m20 += v20[:, None] * u20[None, :]
        m21 += v21[:, None] * u21[None, :]
        m22 += v22[:, None] * u22[None, :]
        m23 += v23[:, None] * u23[None, :]

        m30 += v30[:, None] * u30[None, :]
        m31 += v31[:, None] * u31[None, :]
        m32 += v32[:, None] * u32[None, :]
        m33 += v33[:, None] * u33[None, :]

    # Output transform Y = A * M * A^T
    # A = [[1,1,1,0],[0,1,-1,-1]] (2x4)
    # First: tmp = A * M  (2x4)
    # tmp_row0 = M0 + M1 + M2
    t0_0 = m00 + m10 + m20
    t0_1 = m01 + m11 + m21
    t0_2 = m02 + m12 + m22
    t0_3 = m03 + m13 + m23

    # tmp_row1 = M1 - M2 - M3
    t1_0 = m10 - m20 - m30
    t1_1 = m11 - m21 - m31
    t1_2 = m12 - m22 - m32
    t1_3 = m13 - m23 - m33

    # Then Y = tmp * A^T (2x2)
    # y_row0
    y00 = t0_0 + t0_1 + t0_2
    y01 = t0_1 - t0_2 - t0_3
    # y_row1
    y10 = t1_0 + t1_1 + t1_2
    y11 = t1_1 - t1_2 - t1_3

    # Add bias, divide, leaky_relu
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    bias_vals = bias_vals[None, :]  # (1,BLOCK_N)

    def apply_epilogue(val):
        out = val + bias_vals
        out = out / divisor
        out = tl.where(out >= 0, out, out * negative_slope)
        return out

    y00 = apply_epilogue(y00)
    y01 = apply_epilogue(y01)
    y10 = apply_epilogue(y10)
    y11 = apply_epilogue(y11)

    # Store 2x2 outputs
    b_b = b_idx[:, None]
    co_b = offs_n[None, :]
    oh0_b = oh0[:, None]
    ow0_b = ow0[:, None]

    base_out = (
        y_ptr
        + b_b * stride_y_n
        + co_b * stride_y_c
        + oh0_b * stride_y_h
        + ow0_b * stride_y_w
    )
    mask_out = (mask_m[:, None]) & (mask_n[None, :])

    # top-left (oh0, ow0)
    tl.store(base_out, y00, mask=mask_out)

    # top-right (oh0, ow0+1)
    ptr_tr = base_out + stride_y_w
    tl.store(ptr_tr, y01, mask=mask_out)

    # bottom-left (oh0+1, ow0)
    ptr_bl = base_out + stride_y_h
    tl.store(ptr_bl, y10, mask=mask_out)

    # bottom-right (oh0+1, ow0+1)
    ptr_br = base_out + stride_y_h + stride_y_w
    tl.store(ptr_br, y11, mask=mask_out)


# ----------------------------
# Python wrappers
# ----------------------------

def fused_conv2d_div_leakyrelu(x, weight, bias, divisor, negative_slope=0.01):
    """
    Fused implementation of:
        y = conv2d(x, weight, bias, stride=1, padding=0)
        y = y / divisor
        y = leaky_relu(y, negative_slope)

    Uses Winograd F(2x2,3x3) for 3x3 stride-1 valid convs with even H_out/W_out,
    otherwise falls back to direct GEMM-style kernel.
    """
    assert x.ndim == 4, "Input must be 4D NCHW"
    B, Cin, H_in, W_in = x.shape
    Cout, Cin_w, KH, KW = weight.shape
    assert Cin == Cin_w, "Incompatible input/weight channels"
    assert bias is not None and bias.numel() == Cout

    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    y = torch.empty((B, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_y_n, stride_y_c, stride_y_h, stride_y_w = y.stride()

    # Winograd path: 3x3 kernel, stride=1, padding=0, even output dims
    use_winograd = (
        KH == 3
        and KW == 3
        and (H_out % 2 == 0)
        and (W_out % 2 == 0)
    )

    if use_winograd:
        # Precompute Winograd-transformed weights
        wino_w = precompute_winograd_weight(weight)

        stride_w_cout, stride_w_cin, stride_w_h, stride_w_w = wino_w.stride()

        H_tiles = H_out // 2
        W_tiles = W_out // 2
        total_tiles = B * H_tiles * W_tiles

        BLOCK_M = 16  # tiles
        BLOCK_N = 32  # Cout

        grid = lambda META: (
            triton.cdiv(total_tiles, META["BLOCK_M"]),
            triton.cdiv(Cout, META["BLOCK_N"]),
        )

        winograd_f23_div_leakyrelu_kernel[grid](
            x, wino_w, bias, y,
            B, H_in, W_in,
            H_out, W_out,
            Cin, Cout,
            H_tiles, W_tiles, total_tiles,
            stride_x_n, stride_x_c, stride_x_h, stride_x_w,
            stride_w_cout, stride_w_cin, stride_w_h, stride_w_w,
            stride_y_n, stride_y_c, stride_y_h, stride_y_w,
            float(divisor), float(negative_slope),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return y

    # Fallback: original GEMM-style conv
    M = B * H_out * W_out
    N = Cout
    K = Cin * KH * KW

    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw = weight.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    conv2d_div_leakyrelu_kernel[grid](
        x, weight, bias, y,
        B, H_in, W_in,
        H_out, W_out,
        Cin, Cout,
        KH, KW,
        M, N, K,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        float(divisor), float(negative_slope),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


# ----------------------------
# Module
# ----------------------------

class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
        Conv2d -> divide by constant -> LeakyReLU

    Uses Winograd F(2x2,3x3) for 3x3 stride-1 valid convs with even output size,
    otherwise falls back to a direct GEMM-style Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor

        k = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * k * k
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_conv2d_div_leakyrelu(
            x, self.weight, self.bias, self.divisor, negative_slope=0.01
        )
