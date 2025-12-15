import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_scale_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    KH, KW,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    scale,
    BLOCK_M: tl.constexpr,  # tile over P = N * H_out * W_out
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # tile over K = C_in * KH * KW
):
    # 2D launch grid over output [P, C_out] like a GEMM: [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets along M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Problem sizes
    P = N * H_out * W_out
    K = C_in * KH * KW
    HW_out = H_out * W_out

    # Masks for M and N bounds
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode flattened M index -> (n, oh, ow)
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (C_in * KH * KW), tiled by BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Map flattened K index -> (ic, kh, kw)
        ic = offs_k // (KH * KW)
        rem_k = offs_k % (KH * KW)
        kh = rem_k // KW
        kw = rem_k % KW

        # ------ Load input tile A: shape [BLOCK_M, BLOCK_K] ------
        ih = oh_idx[:, None] + kh[None, :]
        iw = ow_idx[:, None] + kw[None, :]

        x_offsets = (
            n_idx[:, None] * stride_xn
            + ic[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        x_ptrs = x_ptr + x_offsets
        mask_x = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # ------ Load weight tile B: shape [BLOCK_K, BLOCK_N] ------
        w_offsets = (
            offs_n[None, :] * stride_wn
            + ic[:, None] * stride_wc
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
        )
        w_ptrs = w_ptr + w_offsets
        mask_w = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Matrix multiply accumulate: [M,K] x [K,N] -> [M,N]
        acc += tl.dot(x_vals, w_vals)

    # ------ Fuse bias add & scale on the same [M,N] tile ------
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]
    acc = acc * scale

    # ------ Store result ------
    y_offsets = (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    y_ptrs = y_ptr + y_offsets
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask)


@triton.jit
def channel_min_kernel(
    y_ptr, out_ptr,
    N, C_out, H_out, W_out,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # flattened positions (N * H_out * W_out)
):
    # 1D grid over spatial positions P = N * H_out * W_out
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    P = N * H_out * W_out
    mask_m = offs_m < P

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    base_offsets = (
        n_idx * stride_yn
        + oh_idx * stride_yh
        + ow_idx * stride_yw
    )

    # Initialize with +infinity
    current_min = tl.full((BLOCK_M,), 1e30, dtype=tl.float32)

    # Reduction over channel dimension
    for oc in range(0, C_out):
        y_ptrs = y_ptr + base_offsets + oc * stride_yc
        vals = tl.load(y_ptrs, mask=mask_m, other=1e30)
        current_min = tl.minimum(current_min, vals)

    out_offsets = (
        n_idx * stride_on
        + oh_idx * stride_oh
        + ow_idx * stride_ow
    )
    out_ptrs = out_ptr + out_offsets
    tl.store(out_ptrs, current_min, mask=mask_m)


def conv2d_scale_triton(x, weight, bias, scale_factor):
    # x: [N, C_in, H, W]
    # weight: [C_out, C_in, KH, KW]
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight"

    # No padding, stride=1
    H_out = H - KH + 1
    W_out = W - KW + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wkh, stride_wkw = weight.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_scale_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out,
        KH, KW,
        H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wkh, stride_wkw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        scale_factor,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


def conv_scale_min_triton(x, weight, bias, scale_factor):
    # Conv + bias + scale
    y = conv2d_scale_triton(x, weight, bias, scale_factor)
    N, C_out, H_out, W_out = y.shape

    out = torch.empty((N, 1, H_out, W_out), device=y.device, dtype=y.dtype)

    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()
    stride_on, stride_oc, stride_oh, stride_ow = out.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (triton.cdiv(P, meta["BLOCK_M"]),)

    channel_min_kernel[grid](
        y, out,
        N, C_out, H_out, W_out,
        stride_yn, stride_yc, stride_yh, stride_yw,
        stride_on, stride_oc, stride_oh, stride_ow,
        BLOCK_M=256,
        num_warps=4,
        num_stages=1,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton implementation of:
        y = Conv2d(x)
        y = y * scale_factor
        y = min(y, dim=1, keepdim=True)
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        # Expect x: [N, C_in, H, W] on CUDA
        return conv_scale_min_triton(x, self.weight, self.bias, self.scale_factor)
