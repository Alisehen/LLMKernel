import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_gemm_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, K, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile in output channels (C_out)
    BLOCK_N: tl.constexpr,  # tile in output spatial positions (H_out * W_out)
    BLOCK_K: tl.constexpr,  # tile in K dimension (C_in * K * K)
):
    """
    GEMM-style conv2d:
      For each batch n, compute   Y[n] = W @ im2col(X[n]) + b
      where:
        W: [C_out, C_in*K*K]
        im2col(X[n]): [C_in*K*K, H_out*W_out]
        Y[n]: [C_out, H_out*W_out]
    The im2col matrix is generated on the fly inside the kernel.
    """
    pid_n = tl.program_id(0)     # batch index
    pid_m = tl.program_id(1)     # tile over output channels
    pid_p = tl.program_id(2)     # tile over output spatial positions

    # Offsets in output-channel (M) and spatial-position (N) dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_p = pid_p * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    # Masks for bounds
    total_p = H_out * W_out
    mask_m = offs_m < C_out
    mask_p = offs_p < total_p

    # Decode spatial positions (oh, ow) from flattened index p
    oh = offs_p // W_out  # [BN]
    ow = offs_p % W_out   # [BN]

    # Base pointers for this batch element
    x_base = x_ptr + pid_n * stride_xn
    y_base = y_ptr + pid_n * stride_yn

    # K dimension = C_in * K * K
    K_tot = C_in * K * K

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    k0 = 0
    while k0 < K_tot:
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < K_tot

        # Decode offs_k -> (ic, kh, kw)
        kk2 = K * K
        ic = offs_k // kk2
        rem = offs_k % kk2
        kh = rem // K
        kw = rem % K  # all [BK]

        # ---- Load A tile: W[offs_m, offs_k] -> [BM, BK] ----
        # W layout: [C_out, C_in, K, K] with strides stride_wo, stride_wi, stride_wkh, stride_wkw
        w_ptrs = (
            w_ptr
            + offs_m[:, None] * stride_wo
            + ic[None, :] * stride_wi
            + kh[None, :] * stride_wkh
            + kw[None, :] * stride_wkw
        )
        a_mask = mask_m[:, None] & mask_k[None, :]
        A = tl.load(w_ptrs, mask=a_mask, other=0.0)

        # ---- Load B tile: im2col(X[pid_n])[offs_k, offs_p] -> [BK, BN] ----
        # For each spatial position p -> (oh, ow), and each (ic, kh, kw):
        #   hi = oh + kh, wi = ow + kw
        hi = oh[None, :] + kh[:, None]  # [BK, BN]
        wi = ow[None, :] + kw[:, None]  # [BK, BN]

        x_ptrs = (
            x_base
            + ic[:, None] * stride_xc
            + hi * stride_xh
            + wi * stride_xw
        )
        b_mask = mask_k[:, None] & mask_p[None, :]
        B = tl.load(x_ptrs, mask=b_mask, other=0.0)

        # ---- Matmul update ----
        acc += tl.dot(A, B, allow_tf32=True)

        k0 += BLOCK_K

    # Add bias: shape [C_out] -> broadcast to [BM, BN]
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc += bias[:, None]

    # Store results back to Y[n, :, :, :]
    y_oh = oh  # [BN]
    y_ow = ow  # [BN]
    y_ptrs = (
        y_base
        + offs_m[:, None] * stride_yc
        + y_oh[None, :] * stride_yh
        + y_ow[None, :] * stride_yw
    )
    out_mask = mask_m[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def triton_conv2d_nchw(x, weight, bias, kernel_size):
    """
    High-performance GEMM-style conv2d (NCHW, stride=1, no padding).
    Matches torch.nn.functional.conv2d with these settings.
    """
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in_w == C_in
    assert KH == KW == kernel_size
    K = kernel_size

    # Valid conv (no padding, stride=1)
    H_out = H - KH + 1
    W_out = W - KW + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Tiling configuration; all powers of two
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        N,
        triton.cdiv(C_out, META["BLOCK_M"]),
        triton.cdiv(H_out * W_out, META["BLOCK_N"]),
    )

    conv2d_gemm_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, K, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


@triton.jit
def avgpool_sigmoid_sum_kernel(
    x_ptr, out_ptr,
    N, C, H1, W1,
    pool_k, H2, W2,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """
    AvgPool2d (kernel=pool_k, stride=pool_k, no padding) + sigmoid + sum over [C,H2,W2],
    accumulating one scalar per batch element (N).
    """
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    total_hw = H2 * W2
    mask_hw = offs_hw < total_hw
    mask_c = offs_c < C

    oh2 = offs_hw // W2
    ow2 = offs_hw % W2

    oh2_b = oh2[None, :]
    ow2_b = ow2[None, :]
    c_b = offs_c[:, None]

    x_base = x_ptr + pid_n * stride_xn

    pooled_sum = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for ph in range(0, pool_k):
        for pw in range(0, pool_k):
            oh1 = oh2_b * pool_k + ph
            ow1 = ow2_b * pool_k + pw

            x_ptrs = (
                x_base
                + c_b * stride_xc
                + oh1 * stride_xh
                + ow1 * stride_xw
            )

            mask = mask_c[:, None] & mask_hw[None, :]
            vals = tl.load(x_ptrs, mask=mask, other=0.0)
            pooled_sum += vals

    scale = 1.0 / (pool_k * pool_k)
    pooled = pooled_sum * scale

    # Sigmoid
    neg = -pooled
    exp_neg = tl.exp(neg)
    sigmoid = 1.0 / (1.0 + exp_neg)

    # Reduce over C and HW for this tile
    partial = tl.sum(sigmoid, axis=0)  # over C -> [HW]
    partial = tl.sum(partial, axis=0)  # over HW -> scalar

    # Atomic add to per-batch output
    tl.atomic_add(out_ptr + pid_n, partial)


def triton_avgpool_sigmoid_sum(x, pool_kernel_size):
    """
    Wrapper for avgpool + sigmoid + sum over (C,H,W) using Triton.
    """
    x = x.contiguous()
    N, C, H1, W1 = x.shape
    k = pool_kernel_size
    stride = k

    # PyTorch AvgPool2d formula with stride=k, padding=0, dilation=1
    H2 = (H1 - k) // stride + 1
    W2 = (W1 - k) // stride + 1

    out = torch.zeros((N,), device=x.device, dtype=x.dtype)

    BLOCK_C = 32
    BLOCK_HW = 64

    grid = lambda META: (
        N,
        triton.cdiv(H2 * W2, META["BLOCK_HW"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )

    avgpool_sigmoid_sum_kernel[grid](
        x, out,
        N, C, H1, W1,
        k, H2, W2,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model: Conv2d + AvgPool2d + Sigmoid + Sum
    Conv2d is implemented as a GEMM-style Triton kernel with implicit im2col.
    AvgPool2d + Sigmoid + Sum are fused into a second Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Match dtype of parameters (e.g., fp16/bf16/fp32)
        x = x.to(self.weight.dtype)
        conv_out = triton_conv2d_nchw(x, self.weight, self.bias, self.kernel_size)
        out = triton_avgpool_sigmoid_sum(conv_out, self.pool_kernel_size)
        return out
