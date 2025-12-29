import torch, torch.nn as nn, triton, triton.language as tl


# -----------------------------
# Direct GEMM-style Conv Kernel
# (kept as fallback / baseline)
# -----------------------------


@triton.jit
def fused_conv_bn_scale_direct_kernel(
    x_ptr,            # *f32,   [B, Cin, H, W]
    w_ptr,            # *f32,   [Cout, Cin, K, K]
    bias_ptr,         # *f32,   [Cout]
    bn_scale_ptr,     # *f32,   [Cout]  (alpha)
    bn_shift_ptr,     # *f32,   [Cout]  (beta')
    out_ptr,          # *f32,   [B, Cout, Ho, Wo]
    Ho, Wo,           # int32
    M, Kdim,          # int32,  M = B*Ho*Wo, Kdim = Cin*K*K
    Cout, K,          # int32
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,  # tile size in "M" (N * Ho * Wo)
    BLOCK_N: tl.constexpr,  # tile size in Cout
    BLOCK_K: tl.constexpr,  # tile size in Kdim
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < Cout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    nhw = Ho * Wo
    kk = K * K

    # Loop over Kdim (Cin * K * K)
    for k_start in range(0, Kdim, BLOCK_K):
        offs_k = k_start + offs_k_init
        mask_k = offs_k < Kdim

        # ----- A tile (input-im2col) : [BLOCK_M, BLOCK_K] -----
        m_b = offs_m[:, None]
        n_idx = m_b // nhw
        rem_m = m_b - n_idx * nhw
        ho_idx = rem_m // Wo
        wo_idx = rem_m - ho_idx * Wo

        k_a = offs_k[None, :]
        cin_a = k_a // kk
        rem_ka = k_a - cin_a * kk
        kh_a = rem_ka // K
        kw_a = rem_ka - kh_a * K

        h_in = ho_idx + kh_a
        w_in = wo_idx + kw_a

        x_ptrs = (
            x_ptr
            + n_idx * stride_x_n
            + cin_a * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )

        a_mask = mask_m[:, None] & mask_k[None, :]

        a = tl.load(x_ptrs, mask=a_mask, other=0.0)

        # ----- B tile (weights) : [BLOCK_K, BLOCK_N] -----
        k_b = offs_k[:, None]
        out_b = offs_n[None, :]

        cin_b = k_b // kk
        rem_kb = k_b - cin_b * kk
        kh_b = rem_kb // K
        kw_b = rem_kb - kh_b * K

        w_ptrs = (
            w_ptr
            + out_b * stride_w_oc
            + cin_b * stride_w_ic
            + kh_b * stride_w_kh
            + kw_b * stride_w_kw
        )

        b_mask = mask_k[:, None] & mask_n[None, :]

        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Add convolution bias per output channel
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply fused BatchNorm + scaling: y = alpha * conv_out + beta'
    bn_scale = tl.load(bn_scale_ptr + offs_n, mask=mask_n, other=0.0)
    bn_shift = tl.load(bn_shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * bn_scale[None, :] + bn_shift[None, :]

    # Store result to [B, Cout, Ho, Wo]
    m_b = offs_m[:, None]
    n_idx = m_b // nhw
    rem_m = m_b - n_idx * nhw
    ho_idx = rem_m // Wo
    wo_idx = rem_m - ho_idx * Wo

    out_ptrs = (
        out_ptr
        + n_idx * stride_out_n
        + offs_n[None, :] * stride_out_c
        + ho_idx * stride_out_h
        + wo_idx * stride_out_w
    )

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_conv_bn_scale_direct(x, weight, bias, bn_scale, bn_shift, scaling_factor):
    # x:        [B, Cin, H, W]
    # weight:   [Cout, Cin, K, K]
    # bias:     [Cout]
    # bn_scale: [Cout]  (gamma / sqrt(var+eps) * scaling_factor)
    # bn_shift: [Cout]  ((beta - mu*gamma/sqrt(var+eps)) * scaling_factor)
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert x.dtype == torch.float32, "This fused kernel currently supports float32."

    B, Cin, H, W = x.shape
    Cout, Cin_w, K, K_w = weight.shape
    assert Cin == Cin_w and K == K_w, "Incompatible input/weight shapes."

    Ho = H - K + 1
    Wo = W - K + 1
    assert Ho > 0 and Wo > 0, "Input too small for valid convolution with given kernel size."

    x_c = x.contiguous()
    w_c = weight.contiguous()
    bias_c = bias.contiguous()
    bn_scale_c = bn_scale.contiguous()
    bn_shift_c = bn_shift.contiguous()

    out = torch.empty((B, Cout, Ho, Wo), device=x.device, dtype=x.dtype)

    M = B * Ho * Wo
    Kdim = Cin * K * K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(Cout, META["BLOCK_N"]),
    )

    fused_conv_bn_scale_direct_kernel[grid](
        x_c, w_c, bias_c, bn_scale_c, bn_shift_c, out,
        Ho, Wo, M, Kdim, Cout, K,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )

    return out


# -----------------------------
# Winograd F(2x2, 3x3) Conv Kernel
# -----------------------------


def winograd_weight_transform_2x2_3x3(weight: torch.Tensor) -> torch.Tensor:
    """
    Transform 3x3 conv weights to Winograd F(2x2,3x3) domain.

    Input:  weight [Cout, Cin, 3, 3]
    Output: w_winograd [16, Cin, Cout]
        index 0..15 corresponds to 4x4 transformed tile (row-major)
    """
    assert weight.ndim == 4 and weight.shape[2] == 3 and weight.shape[3] == 3
    Cout, Cin, Kh, Kw = weight.shape
    device = weight.device
    dtype = weight.dtype

    # G matrix (4x3) from Lavin, "Fast Algorithms for Convolutional Neural Networks"
    G = torch.tensor(
        [
            [1.0 / 4.0,      0.0,       0.0],
            [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
            [-1.0 / 6.0,  1.0 / 6.0, -1.0 / 6.0],
            [1.0 / 24.0, 1.0 / 12.0,  1.0 / 6.0],
        ],
        device=device,
        dtype=dtype,
    )  # shape (4,3)

    g = weight.reshape(Cout * Cin, 3, 3)  # [B,3,3], B=Cout*Cin
    # G @ g @ G^T -> [B,4,4]
    temp = torch.matmul(G, g)              # [B,4,3]
    g_trans = torch.matmul(temp, G.t())    # [B,4,4]
    g_trans = g_trans.reshape(Cout, Cin, 16)  # [Cout, Cin, 16]
    # Reorder to [16, Cin, Cout] with contiguous memory
    w_winograd = g_trans.permute(2, 1, 0).contiguous()
    return w_winograd


@triton.jit
def winograd_f2x2_conv_bn_scale_kernel(
    x_ptr,                  # *f32, [B, Cin, H, W]
    w_winograd_ptr,         # *f32, [16, Cin, Cout] in row-major
    bias_ptr,               # *f32, [Cout]
    bn_scale_ptr,           # *f32, [Cout]
    bn_shift_ptr,           # *f32, [Cout]
    out_ptr,                # *f32, [B, Cout, Ho, Wo]
    B, Cin, H, W, Ho, Wo,   # int32
    tiles_h, tiles_w,       # int32: Ho/2, Wo/2
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_ww_p, stride_ww_cin, stride_ww_cout,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_OC: tl.constexpr,  # tile in Cout
    BLOCK_IC: tl.constexpr,  # chunk in Cin
):
    # program ids
    pid_tile = tl.program_id(0)
    pid_oc = tl.program_id(1)

    tiles_per_batch = tiles_h * tiles_w

    n = pid_tile // tiles_per_batch
    rem = pid_tile - n * tiles_per_batch
    ty = rem // tiles_w
    tx = rem - ty * tiles_w

    # top-left output index for this 2x2 tile
    oh0 = ty * 2
    ow0 = tx * 2

    # output channel block
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < (stride_ww_cout * 0 + tl.uint32(0) + 0)  # placeholder, overwritten below (avoid unused warning)

    # We'll reassign oc_mask properly after loading Cout from strides
    # but Triton requires oc_mask variable to be defined before first use.
    # We don't use this placeholder value.

    Cout = stride_ww_cout  # we pass Cout as stride_ww_cout's dedicated arg? No.
    # The above line is incorrect; we don't actually encode Cout in strides.
    # To avoid confusion and ensure correctness, Cout is not recovered from strides.
    # Instead, we rely only on bounds via pointer arithmetic masks.
    # Proper oc_mask will be defined based on Ho/Wo/Cin/Cout constraints outside.

    # ------- Correct reconstruction for oc_mask --------
    # We don't know Cout inside the kernel directly, but we can infer it
    # by passing it via one of the strides. To avoid ABI breakage, we
    # encode Cout in stride_ww_p (which is otherwise independent).
    Cout = stride_ww_p  # reinterpret stride_ww_p as Cout (see launcher)
    oc_mask = offs_oc < Cout

    # accumulators in Winograd domain: [16, BLOCK_OC]
    acc = tl.zeros((16, BLOCK_OC), dtype=tl.float32)

    # Loop over Cin in chunks
    for ic_start in range(0, Cin, BLOCK_IC):
        offs_ci = ic_start + tl.arange(0, BLOCK_IC)
        ci_mask = offs_ci < Cin

        # Load 4x4 input patch for each Cin chunk, and apply input Winograd transform
        # d: [16, BLOCK_IC]
        d = tl.zeros((16, BLOCK_IC), dtype=tl.float32)

        # Load spatial 4x4 patch per input channel
        for r in range(4):
            h_in = oh0 + r
            for c in range(4):
                w_in = ow0 + c
                p = r * 4 + c
                x_ptrs = (
                    x_ptr
                    + n * stride_x_n
                    + offs_ci * stride_x_c
                    + h_in * stride_x_h
                    + w_in * stride_x_w
                )
                d[p, :] = tl.load(x_ptrs, mask=ci_mask, other=0.0)

        # Winograd input transform V = B^T d B
        # First: row transform (B^T)
        T = tl.zeros((16, BLOCK_IC), dtype=tl.float32)

        for r in range(4):
            p0 = r * 4 + 0
            p1 = r * 4 + 1
            p2 = r * 4 + 2
            p3 = r * 4 + 3
            d0 = d[p0, :]
            d1 = d[p1, :]
            d2 = d[p2, :]
            d3 = d[p3, :]

            # B^T rows:
            # [1, 0, -1, 0]
            # [0, 1,  1, 0]
            # [0,-1,  1, 0]
            # [0, 1,  0,-1]
            T[p0, :] = d0 - d2
            T[p1, :] = d1 + d2
            T[p2, :] = -d1 + d2
            T[p3, :] = d1 - d3

        # Second: column transform (B)
        V = tl.zeros((16, BLOCK_IC), dtype=tl.float32)
        for c in range(4):
            i0 = 0 * 4 + c
            i1 = 1 * 4 + c
            i2 = 2 * 4 + c
            i3 = 3 * 4 + c
            t0 = T[i0, :]
            t1 = T[i1, :]
            t2 = T[i2, :]
            t3 = T[i3, :]

            V[i0, :] = t0 - t2
            V[i1, :] = t1 + t2
            V[i2, :] = -t1 + t2
            V[i3, :] = t1 - t3

        # Element-wise multiplication + reduction over Cin:
        # For each Winograd position p (0..15):
        #   acc[p, oc] += sum_ci V[p,ci] * W[p,ci,oc]
        for p in range(16):
            v_row = V[p, :]  # [BLOCK_IC]

            v_mat = v_row[None, :]  # [1, BLOCK_IC]

            # Weight tile: [BLOCK_IC, BLOCK_OC] from w_winograd[p, offs_ci, offs_oc]
            w_ptrs = (
                w_winograd_ptr
                + p * stride_ww_p  # stride_ww_p is Cout: see launcher documentation below
                + offs_ci[:, None] * stride_ww_cin
                + offs_oc[None, :] * stride_ww_cout
            )
            w = tl.load(w_ptrs, mask=(ci_mask[:, None] & oc_mask[None, :]), other=0.0)

            prod = tl.dot(v_mat, w, allow_tf32=True)  # [1, BLOCK_OC]
            acc[p, :] += prod[0, :]

    # Now acc holds M (Winograd output) for a 4x4 tile:
    # reshape implicit: p -> (r,c) with r,c in 0..3

    # Output Winograd transform: Y = A^T M A
    # A^T = [[1, 1, 1, 0],
    #        [0, 1,-1,-1]]

    # Row transform: T = A^T M  -> shape [2,4]
    # We materialize as t0_c*, t1_c* (each [BLOCK_OC])
    m00 = acc[0, :]
    m10 = acc[4, :]
    m20 = acc[8, :]
    m30 = acc[12, :]

    m01 = acc[1, :]
    m11 = acc[5, :]
    m21 = acc[9, :]
    m31 = acc[13, :]

    m02 = acc[2, :]
    m12 = acc[6, :]
    m22 = acc[10, :]
    m32 = acc[14, :]

    m03 = acc[3, :]
    m13 = acc[7, :]
    m23 = acc[11, :]
    m33 = acc[15, :]

    t0_c0 = m00 + m10 + m20
    t1_c0 = m10 - m20 - m30

    t0_c1 = m01 + m11 + m21
    t1_c1 = m11 - m21 - m31

    t0_c2 = m02 + m12 + m22
    t1_c2 = m12 - m22 - m32

    t0_c3 = m03 + m13 + m23
    t1_c3 = m13 - m23 - m33

    # Column transform: Y = T A -> 2x2 outputs
    y00 = t0_c0 + t0_c1 + t0_c2
    y01 = t0_c1 - t0_c2 - t0_c3
    y10 = t1_c0 + t1_c1 + t1_c2
    y11 = t1_c1 - t1_c2 - t1_c3

    # Load bias and BN fusion params
    bias = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)
    bn_scale = tl.load(bn_scale_ptr + offs_oc, mask=oc_mask, other=0.0)
    bn_shift = tl.load(bn_shift_ptr + offs_oc, mask=oc_mask, other=0.0)

    # Apply: out = (conv_out + bias) * bn_scale + bn_shift
    y00 = (y00 + bias) * bn_scale + bn_shift
    y01 = (y01 + bias) * bn_scale + bn_shift
    y10 = (y10 + bias) * bn_scale + bn_shift
    y11 = (y11 + bias) * bn_scale + bn_shift

    # Store results to out[N, Cout, Ho, Wo]
    out_base = out_ptr + n * stride_out_n + offs_oc * stride_out_c

    ptr00 = out_base + (oh0 + 0) * stride_out_h + (ow0 + 0) * stride_out_w
    ptr01 = out_base + (oh0 + 0) * stride_out_h + (ow0 + 1) * stride_out_w
    ptr10 = out_base + (oh0 + 1) * stride_out_h + (ow0 + 0) * stride_out_w
    ptr11 = out_base + (oh0 + 1) * stride_out_h + (ow0 + 1) * stride_out_w

    tl.store(ptr00, y00, mask=oc_mask)
    tl.store(ptr01, y01, mask=oc_mask)
    tl.store(ptr10, y10, mask=oc_mask)
    tl.store(ptr11, y11, mask=oc_mask)


def fused_conv_bn_scale_winograd(x, weight, bias, bn_scale, bn_shift, scaling_factor):
    """
    Winograd-accelerated fused Conv+BN+scale.

    Falls back to direct Triton kernel if Winograd preconditions are not met:
      - kernel_size != 3
      - stride != 1, padding != 0, dilation != 1 (enforced in ModelNew)
      - Ho or Wo <= 0
      - Ho or Wo not divisible by 2
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert x.dtype == torch.float32, "This fused kernel currently supports float32."

    B, Cin, H, W = x.shape
    Cout, Cin_w, K, K_w = weight.shape
    assert Cin == Cin_w and K == K_w, "Incompatible input/weight shapes."

    Ho = H - K + 1
    Wo = W - K + 1
    assert Ho > 0 and Wo > 0, "Input too small for valid convolution with given kernel size."

    use_winograd = (
        (K == 3)
        and (Ho % 2 == 0)
        and (Wo % 2 == 0)
        and (Ho >= 2)
        and (Wo >= 2)
    )

    if not use_winograd:
        return fused_conv_bn_scale_direct(x, weight, bias, bn_scale, bn_shift, scaling_factor)

    # Prepare contiguous layouts
    x_c = x.contiguous()
    w_c = weight.contiguous()
    bias_c = bias.contiguous()
    bn_scale_c = bn_scale.contiguous()
    bn_shift_c = bn_shift.contiguous()

    # Precompute Winograd-transformed weights: [16, Cin, Cout]
    w_winograd = winograd_weight_transform_2x2_3x3(w_c)
    w_winograd_c = w_winograd.contiguous()

    out = torch.empty((B, Cout, Ho, Wo), device=x.device, dtype=x.dtype)

    tiles_h = Ho // 2
    tiles_w = Wo // 2
    num_tiles = B * tiles_h * tiles_w

    # We'll encode Cout in stride_ww_p to make it visible to the kernel
    stride_ww_p, stride_ww_cin, stride_ww_cout = w_winograd_c.stride()

    grid = lambda META: (
        num_tiles,
        triton.cdiv(Cout, META["BLOCK_OC"]),
    )

    winograd_f2x2_conv_bn_scale_kernel[grid](
        x_c,
        w_winograd_c,
        bias_c,
        bn_scale_c,
        bn_shift_c,
        out,
        B, Cin, H, W, Ho, Wo,
        tiles_h, tiles_w,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        Cout,                 # overload: stride_ww_p used as Cout in kernel
        stride_ww_cin,
        stride_ww_cout,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_OC=64,
        BLOCK_IC=16,
        num_warps=4,
        num_stages=2,
    )

    return out


# -----------------------------
# High-level Module
# -----------------------------


class ModelNew(nn.Module):
    """
    Fused Triton implementation of:
      Conv2d -> BatchNorm2d (inference) -> scaling_factor

    Uses Winograd F(2x2,3x3) convolution for 3x3, stride=1, padding=0,
    with even spatial output sizes; otherwise falls back to a direct
    GEMM-style Triton convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        assert x.ndim == 4, "Input must be NCHW"
        assert self.conv.stride == (1, 1), "Kernel only supports stride=1"
        assert self.conv.padding == (0, 0), "Kernel only supports padding=0"
        assert self.conv.dilation == (1, 1), "Kernel only supports dilation=1"
        assert self.conv.groups == 1, "Kernel only supports groups=1"

        weight = self.conv.weight
        if self.conv.bias is None:
            bias = torch.zeros(
                weight.size(0),
                device=weight.device,
                dtype=weight.dtype,
            )
        else:
            bias = self.conv.bias

        bn = self.bn
        running_mean = bn.running_mean.to(weight.device, dtype=weight.dtype)
        running_var = bn.running_var.to(weight.device, dtype=weight.dtype)
        gamma = bn.weight.to(weight.device, dtype=weight.dtype)
        beta = bn.bias.to(weight.device, dtype=weight.dtype)
        eps = bn.eps

        # Precompute fused BatchNorm (inference) + scaling parameters
        # out = scaling_factor * [ (x - mu)/sqrt(var+eps) * gamma + beta ]
        inv_std = torch.rsqrt(running_var + eps)
        bn_scale = gamma * inv_std * self.scaling_factor
        bn_shift = (beta - running_mean * inv_std * gamma) * self.scaling_factor

        return fused_conv_bn_scale_winograd(
            x, weight, bias, bn_scale, bn_shift, self.scaling_factor
        )
