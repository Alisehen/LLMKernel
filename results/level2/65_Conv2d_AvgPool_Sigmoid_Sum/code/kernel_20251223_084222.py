import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    KH, KW, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over (N*OH*OW)
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # tile over K = C_in*KH*KW
):
    # Program IDs for tiling over output matrix [M, C_out] with M=N*OH*OW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * OH * OW
    K_tot = C_in * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode linear M index -> (n_idx, oh_idx, ow_idx)
    hw = OH * OW
    n_idx = offs_m // hw
    rem = offs_m % hw
    oh_idx = rem // OW
    ow_idx = rem % OW

    # Accumulator in fp32 for better numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (flattened C_in*KH*KW)
    for k0 in range(0, K_tot, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_tot

        # Decode flattened K index -> (ic, kh, kw)
        khw = KH * KW
        ic = offs_k // khw
        rem_k = offs_k % khw
        kh = rem_k // KW
        kw = rem_k % KW

        # Build input pointers: shape (BLOCK_M, BLOCK_K)
        n_b = n_idx[:, None]
        oh_b = oh_idx[:, None]
        ow_b = ow_idx[:, None]
        ic_b = ic[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        h_in = oh_b + kh_b
        w_in = ow_b + kw_b

        a_ptrs = (
            x_ptr
            + n_b * stride_xn
            + ic_b * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )

        # Build weight pointers: shape (BLOCK_K, BLOCK_N)
        oc_b = offs_n[None, :]
        ic_w = ic[:, None]
        kh_w = kh[:, None]
        kw_w = kw[:, None]

        w_ptrs = (
            w_ptr
            + oc_b * stride_wo
            + ic_w * stride_wc
            + kh_w * stride_wkh
            + kw_w * stride_wkw
        )

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias: shape (C_out,)
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store result to y[n, oc, oh, ow]
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv2d_triton(x, weight, bias):
    """
    NCHW conv2d with stride=1, padding=0, dilation=1, groups=1,
    using an implicit-im2col GEMM in Triton.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_w, KH, KW = weight.shape
    assert C_w == C_in
    assert KH == KW  # model uses square kernels

    OH = H - KH + 1
    OW = W - KW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(N * OH * OW, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out,
        KH, KW, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return y


@triton.jit
def avgpool_sigmoid_sum_kernel(
    in_ptr, out_ptr,
    N, C, H, W,
    pool_ksize, stride_p,
    Hp, Wp,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    BLOCK_D: tl.constexpr,
):
    # One program per batch element
    pid_n = tl.program_id(0)
    n = pid_n

    # Total number of pooled elements per sample across C, Hp, Wp
    D = C * Hp * Wp

    total = tl.zeros((1,), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D

        # Decode offs -> (c_idx, ph, pw)
        hw = Hp * Wp
        c_idx = offs // hw
        rem = offs % hw
        ph = rem // Wp
        pw = rem % Wp

        win_sum = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Average pooling (kernel=pool_ksize, stride=stride_p)
        for kh in range(0, pool_ksize):
            for kw in range(0, pool_ksize):
                h_in = ph * stride_p + kh
                w_in = pw * stride_p + kw

                in_ptrs = (
                    in_ptr
                    + n * stride_in_n
                    + c_idx * stride_in_c
                    + h_in * stride_in_h
                    + w_in * stride_in_w
                )
                vals = tl.load(in_ptrs, mask=mask, other=0.0)
                win_sum += vals

        area = pool_ksize * pool_ksize
        avg = win_sum / area

        # Sigmoid: 1 / (1 + exp(-x))
        s = 1.0 / (1.0 + tl.exp(-avg))

        # Zero-out invalid lanes
        s = tl.where(mask, s, 0.0)

        block_sum = tl.sum(s, axis=0)
        total += block_sum

    tl.store(out_ptr + n, total)


def avgpool_sigmoid_sum_triton(x, pool_kernel_size: int):
    """
    x: (N, C, H, W) from conv2d (no padding, stride 1).
    Applies AvgPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size),
    then sigmoid, then sums over (C,Hp,Wp) per batch element.
    """
    assert x.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    k = int(pool_kernel_size)
    stride_p = k

    Hp = (H - k) // stride_p + 1
    Wp = (W - k) // stride_p + 1

    out = torch.empty((N,), device=x.device, dtype=x.dtype)

    BLOCK_D = 128

    grid = lambda META: (max(1, N),)

    avgpool_sigmoid_sum_kernel[grid](
        x, out,
        N, C, H, W,
        k, stride_p,
        Hp, Wp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of:

        Conv2d -> AvgPool2d -> Sigmoid -> Sum over C,H,W

    The conv and pooling+sigmoid+sum are implemented with high-performance
    Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        k = int(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.pool_kernel_size = int(pool_kernel_size)

        # Initialize weights similar to nn.Conv2d defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure data is on same device as parameters
        x = x.to(self.weight.device)
        x = conv2d_triton(x, self.weight, self.bias)
        x = avgpool_sigmoid_sum_triton(x, self.pool_kernel_size)
        return x
