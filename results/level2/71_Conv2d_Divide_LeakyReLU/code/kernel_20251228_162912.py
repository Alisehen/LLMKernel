import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_div_leakyrelu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    divisor, negative_slope,
    N, C_in,
    H_in, W_in,
    C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K_total,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Baseline implicit-GEMM 2D convolution with fused divide and LeakyReLU.

    x:      (N, C_in, H_in, W_in)
    w:      (C_out, C_in, KERNEL_SIZE, KERNEL_SIZE)
    bias:   (C_out,)
    y:      (N, C_out, H_out, W_out), where H_out = H_in - KERNEL_SIZE + 1, similarly for W_out.
    """

    pid_m = tl.program_id(0)  # along flattened output positions (N * H_out * W_out)
    pid_n = tl.program_id(1)  # along output channels

    # Offsets in the flattened M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * H_out * W_out

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode flattened output index m -> (n_idx, oh_idx, ow_idx)
    tmp = offs_m
    hw_out = H_out * W_out
    n_idx = tmp // hw_out
    tmp = tmp % hw_out
    oh_idx = tmp // W_out
    ow_idx = tmp % W_out

    # Prepare K offsets
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in FP32 for better numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KS = KERNEL_SIZE
    KS2 = KS * KS

    for k_start in range(0, K_total, BLOCK_K):
        k_offsets = k_start + offs_k  # [BLOCK_K]
        mask_k = k_offsets < K_total

        # Map flattened k index -> (c_idx, kh_idx, kw_idx)
        c_idx = k_offsets // KS2
        rem = k_offsets % KS2
        kh_idx = rem // KS
        kw_idx = rem % KS

        # Broadcast indices for input (A matrix: [M, K])
        n_b = n_idx[:, None]
        oh_b = oh_idx[:, None]
        ow_b = ow_idx[:, None]

        c_b = c_idx[None, :]
        kh_b = kh_idx[None, :]
        kw_b = kw_idx[None, :]

        # Compute input pointers
        x_ptrs = (
            x_ptr
            + n_b * stride_xn
            + c_b * stride_xc
            + (oh_b + kh_b) * stride_xh
            + (ow_b + kw_b) * stride_xw
        )

        # Bounds mask for input (for generality; always true for valid conv w/o padding)
        in_bounds_h = (oh_b + kh_b) < H_in
        in_bounds_w = (ow_b + kw_b) < W_in
        mask_hw = in_bounds_h & in_bounds_w

        mask_a = mask_m[:, None] & mask_k[None, :] & mask_hw

        a = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # Weight (B matrix: [K, C_out])
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wn
            + c_idx[:, None] * stride_wc
            + kh_idx[:, None] * stride_wk
            + kw_idx[:, None] * stride_wl
        )

        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptrs, mask=mask_b, other=0.0)

        # Accumulate in FP32
        a_fp32 = a.to(tl.float32)
        b_fp32 = b.to(tl.float32)
        acc += tl.dot(a_fp32, b_fp32, allow_tf32=True)

    # Add bias
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Divide by constant
    inv_div = 1.0 / divisor
    acc = acc * inv_div

    # LeakyReLU
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store to output
    n_b = n_idx[:, None]
    oh_b = oh_idx[:, None]
    ow_b = ow_idx[:, None]

    y_ptrs = (
        y_ptr
        + n_b * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_b * stride_yh
        + ow_b * stride_yw
    )
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


def conv2d_div_leakyrelu(x, weight, bias, divisor, negative_slope=0.01):
    """
    Baseline Triton kernel implementing:
      y = LeakyReLU( conv2d(x, weight, bias) / divisor )
    for stride=1, padding=0, dilation=1, groups=1.
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C_in, H_in, W_in = x.shape
    C_out, Cw_in, KH, KW = weight.shape
    assert C_in == Cw_in, "Incompatible in_channels between input and weight"
    assert KH == KW, "Only square kernels are supported"
    KS = KH

    # Valid convolution: no padding, stride 1
    H_out = H_in - KS + 1
    W_out = W_in - KS + 1
    assert H_out > 0 and W_out > 0, "Kernel larger than input with no padding"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    K_total = C_in * KS * KS

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_div_leakyrelu_kernel[grid](
        x, weight, bias, y,
        float(divisor), float(negative_slope),
        N, C_in,
        H_in, W_in,
        C_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        K_total,
        KERNEL_SIZE=KS,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return y


# ===== Winograd F(2x2, 3x3) implementation =====


def winograd_weight_transform_f2k3(weight: torch.Tensor) -> torch.Tensor:
    """
    Transform 3x3 convolution weights into Winograd F(2x2,3x3) domain:
      U = G * g * G^T

    weight: (C_out, C_in, 3, 3)
    returns: (C_out, C_in, 4, 4)
    """
    C_out, C_in, KH, KW = weight.shape
    assert KH == 3 and KW == 3, "Winograd path only supports 3x3 kernels"

    G = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=weight.dtype,
        device=weight.device,
    )
    Gt = G.t()

    w_reshaped = weight.reshape(-1, 3, 3)  # (C_out*C_in, 3, 3)
    # U = G * w * G^T (batched)
    U = torch.einsum("ij,bjk,kl->bil", G, w_reshaped, Gt)  # (B,4,4)
    U = U.reshape(C_out, C_in, 4, 4)
    return U


@triton.jit
def conv2d_winograd_f2k3_div_leakyrelu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    divisor, negative_slope,
    N, H_in, W_in,
    C_out, H_out, W_out,
    tiles_h, tiles_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wp, stride_wq,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C_IN: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Winograd F(2x2,3x3) convolution with fused divide and LeakyReLU.

    x:          (N, C_in, H_in, W_in)
    w (Winograd): (C_out, C_in, 4, 4) = G * g * G^T
    bias:       (C_out,)
    y:          (N, C_out, H_out, W_out) where H_out = H_in - 2, W_out = W_in - 2

    Each kernel instance computes one 2x2 output tile for a block of output channels.
    """

    pid_tile = tl.program_id(0)   # over (N, tiles_h, tiles_w)
    pid_oc_blk = tl.program_id(1)  # over output-channel blocks

    tiles_per_img = tiles_h * tiles_w
    n = pid_tile // tiles_per_img
    tile_in_img = pid_tile % tiles_per_img
    th = tile_in_img // tiles_w
    tw = tile_in_img % tiles_w

    # Base output coordinates for this 2x2 tile
    oh_base = th * 2
    ow_base = tw * 2

    # Output channel block
    oc_start = pid_oc_blk * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < C_out

    # Accumulators in Winograd domain: 4 rows x 4 cols per output channel block
    acc0 = tl.zeros((BLOCK_OC, 4), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_OC, 4), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_OC, 4), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_OC, 4), dtype=tl.float32)

    rs = tl.arange(0, 4)
    cs = tl.arange(0, 4)

    # Loop over input channels (compile-time constant)
    for ci in range(0, C_IN):
        # Input 4x4 patch for this tile and input channel
        h_idx = oh_base + rs  # (4,)
        w_idx = ow_base + cs  # (4,)

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + ci * stride_xc
            + h_idx[:, None] * stride_xh
            + w_idx[None, :] * stride_xw
        )

        mask_h = h_idx[:, None] < H_in
        mask_w = w_idx[None, :] < W_in
        mask_x = mask_h & mask_w

        d = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)  # (4,4)

        # Winograd input transform: V = B^T * d * B
        # B^T = [[1,0,-1,0],
        #        [0,1, 1,0],
        #        [0,-1,1,0],
        #        [0,1, 0,-1]]
        # Step 1: rows: tmp = B^T * d
        r0 = d[0, :]
        r1 = d[1, :]
        r2 = d[2, :]
        r3 = d[3, :]

        t0 = r0 - r2
        t1 = r1 + r2
        t2 = -r1 + r2
        t3 = r1 - r3

        # Step 2: columns: V = tmp * B  (B is symmetric to B^T)
        v0 = t0 - t2
        v1 = t1 + t2
        v2 = -t1 + t2
        v3 = t1 - t3
        # v0..v3 are row vectors of shape (4,)

        # Load transformed weights U for this input channel & output-channel block
        p = tl.arange(0, 4)
        q = tl.arange(0, 4)
        w_ptrs = (
            w_ptr
            + oc_offsets[:, None, None] * stride_wn
            + ci * stride_wc
            + p[None, :, None] * stride_wp
            + q[None, None, :] * stride_wq
        )
        w_mask = mask_oc[:, None, None]
        U = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # (BLOCK_OC,4,4)

        U0 = U[:, 0, :]  # (BLOCK_OC,4)
        U1 = U[:, 1, :]
        U2 = U[:, 2, :]
        U3 = U[:, 3, :]

        acc0 += U0 * v0[None, :]
        acc1 += U1 * v1[None, :]
        acc2 += U2 * v2[None, :]
        acc3 += U3 * v3[None, :]

    # Inverse Winograd transform: Y = A^T * M * A
    # A^T = [[1,1,1,0],
    #        [0,1,-1,-1]]
    m0 = acc0
    m1 = acc1
    m2 = acc2
    m3 = acc3

    # Row transform: T = A^T * M  -> (2 x 4)
    t0 = m0 + m1 + m2         # (BLOCK_OC,4)
    t1 = m1 - m2 - m3         # (BLOCK_OC,4)

    # Column transform: Y = T * A  -> (2 x 2) outputs
    t0_c0 = t0[:, 0]
    t0_c1 = t0[:, 1]
    t0_c2 = t0[:, 2]
    t0_c3 = t0[:, 3]

    t1_c0 = t1[:, 0]
    t1_c1 = t1[:, 1]
    t1_c2 = t1[:, 2]
    t1_c3 = t1[:, 3]

    y00 = t0_c0 + t0_c1 + t0_c2
    y01 = t0_c1 - t0_c2 - t0_c3
    y10 = t1_c0 + t1_c1 + t1_c2
    y11 = t1_c1 - t1_c2 - t1_c3

    # Bias, divide, LeakyReLU
    bias = tl.load(bias_ptr + oc_offsets, mask=mask_oc, other=0.0)
    inv_div = 1.0 / divisor

    y00 = y00 + bias
    y01 = y01 + bias
    y10 = y10 + bias
    y11 = y11 + bias

    y00 = y00 * inv_div
    y01 = y01 * inv_div
    y10 = y10 * inv_div
    y11 = y11 * inv_div

    y00 = tl.where(y00 >= 0, y00, y00 * negative_slope)
    y01 = tl.where(y01 >= 0, y01, y01 * negative_slope)
    y10 = tl.where(y10 >= 0, y10, y10 * negative_slope)
    y11 = tl.where(y11 >= 0, y11, y11 * negative_slope)

    # Store results to y (N, C_out, H_out, W_out)
    y_base_n = n * stride_yn
    oc_base = oc_offsets * stride_yc

    oh0 = oh_base + 0
    oh1 = oh_base + 1
    ow0 = ow_base + 0
    ow1 = ow_base + 1

    ptr00 = y_ptr + y_base_n + oc_base + oh0 * stride_yh + ow0 * stride_yw
    ptr01 = y_ptr + y_base_n + oc_base + oh0 * stride_yh + ow1 * stride_yw
    ptr10 = y_ptr + y_base_n + oc_base + oh1 * stride_yh + ow0 * stride_yw
    ptr11 = y_ptr + y_base_n + oc_base + oh1 * stride_yh + ow1 * stride_yw

    mask00 = mask_oc & (oh0 < H_out) & (ow0 < W_out)
    mask01 = mask_oc & (oh0 < H_out) & (ow1 < W_out)
    mask10 = mask_oc & (oh1 < H_out) & (ow0 < W_out)
    mask11 = mask_oc & (oh1 < H_out) & (ow1 < W_out)

    tl.store(ptr00, y00, mask=mask00)
    tl.store(ptr01, y01, mask=mask01)
    tl.store(ptr10, y10, mask=mask10)
    tl.store(ptr11, y11, mask=mask11)


def conv2d_winograd_f2k3_div_leakyrelu(x, weight_winograd, bias, divisor, negative_slope=0.01):
    """
    Winograd F(2x2,3x3) wrapper:
      y = LeakyReLU( conv2d(x, weight, bias) / divisor )
    using pre-transformed weights (G * w * G^T).

    x:                (N, C_in, H_in, W_in)
    weight_winograd:  (C_out, C_in, 4, 4)
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C_in, H_in, W_in = x.shape
    C_out, Cw_in, HP, WP = weight_winograd.shape
    assert C_in == Cw_in, "Incompatible in_channels between input and weight"
    assert HP == 4 and WP == 4, "Winograd weights must be 4x4 tiles"

    # Valid convolution: 3x3, stride 1, padding 0
    H_out = H_in - 2
    W_out = W_in - 2
    assert H_out > 0 and W_out > 0, "Kernel larger than input with no padding"

    # F(2x2,3x3) produces 2x2 outputs per tile
    assert H_out % 2 == 0 and W_out % 2 == 0, "Winograd path requires even H_out and W_out"
    tiles_h = H_out // 2
    tiles_w = W_out // 2

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32

    grid = (
        N * tiles_h * tiles_w,
        triton.cdiv(C_out, BLOCK_OC),
    )

    conv2d_winograd_f2k3_div_leakyrelu_kernel[grid](
        x, weight_winograd, bias, y,
        float(divisor), float(negative_slope),
        N, H_in, W_in,
        C_out, H_out, W_out,
        tiles_h, tiles_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight_winograd.stride(0), weight_winograd.stride(1),
        weight_winograd.stride(2), weight_winograd.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        C_IN=C_in,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      Conv2d -> divide by constant -> LeakyReLU
    (stride=1, padding=0, dilation=1, groups=1)

    Uses Winograd F(2x2,3x3) when possible, falls back to baseline kernel otherwise.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels supported"
            k = kernel_size[0]
        else:
            k = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.divisor = float(divisor)
        self.negative_slope = 0.01
        self.kernel_size = k

    def forward(self, x):
        # Winograd path: 3x3, stride=1, padding=0, dilation=1, groups=1
        use_winograd = (
            self.kernel_size == 3
            and x.shape[2] >= 3
            and x.shape[3] >= 3
        )
        if use_winograd:
            N, C_in, H_in, W_in = x.shape
            H_out = H_in - 2
            W_out = W_in - 2
            # Require even output sizes for pure F(2x2,3x3) tiling
            if H_out > 0 and W_out > 0 and (H_out % 2 == 0) and (W_out % 2 == 0):
                w_win = winograd_weight_transform_f2k3(self.weight)
                return conv2d_winograd_f2k3_div_leakyrelu(
                    x, w_win, self.bias, self.divisor, self.negative_slope
                )

        # Fallback to baseline implicit-GEMM Triton conv
        return conv2d_div_leakyrelu(
            x, self.weight, self.bias, self.divisor, self.negative_slope
        )
