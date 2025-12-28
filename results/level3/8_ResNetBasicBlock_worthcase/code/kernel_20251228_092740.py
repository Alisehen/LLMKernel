import torch
import torch.nn as nn
import triton
import triton.language as tl


def fold_conv_bn_2d(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d into Conv2d for inference:
      y = BN(conv(x)) == conv_fused(x)

    This uses BN's running_mean / running_var (i.e., eval-mode behavior).
    """
    W = conv.weight
    if conv.bias is not None:
        b = conv.bias
    else:
        b = torch.zeros(W.shape[0], device=W.device, dtype=W.dtype)

    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    if bn.weight is not None:
        gamma = bn.weight
    else:
        gamma = torch.ones_like(running_mean, device=running_mean.device, dtype=running_mean.dtype)
    if bn.bias is not None:
        beta = bn.bias
    else:
        beta = torch.zeros_like(running_mean, device=running_mean.device, dtype=running_mean.dtype)

    inv_std = torch.rsqrt(running_var + eps)  # 1 / sqrt(var + eps)

    # Fuse: W_fused[o] = W[o] * (gamma[o] * inv_std[o])
    scale = (gamma * inv_std).reshape(-1, 1, 1, 1)
    W_fused = W * scale

    # b_fused[o] = beta[o] + (b[o] - mean[o]) * inv_std[o] * gamma[o]
    b_fused = beta + (b - running_mean) * inv_std * gamma

    return W_fused, b_fused


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    stride_h_conv, stride_w_conv,
    pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wk, stride_wl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over output M dimension: N * H_out * W_out
    BLOCK_N: tl.constexpr,  # tile over output channels (C_out)
    BLOCK_K: tl.constexpr,  # tile over reduction dimension K = C_in * KH * KW
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * H_out * W_out
    K = C_in * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    hw = H_out * W_out
    n_idx = offs_m // hw
    hw_idx = offs_m % hw
    oh = hw_idx // W_out
    ow = hw_idx % W_out

    y_ptrs = y_ptr + (
        n_idx[:, None] * stride_yn +
        offs_n[None, :] * stride_yc +
        oh[:, None] * stride_yh +
        ow[:, None] * stride_yw
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        ci = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        ih = oh[:, None] * stride_h_conv + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w_conv + kw[None, :] - pad_w

        mask_in = (
            mask_m[:, None] & mask_k[None, :] &
            (ci[None, :] < C_in) &
            (ih >= 0) & (ih < H_in) &
            (iw >= 0) & (iw < W_in)
        )

        x_ptrs = x_ptr + (
            n_idx[:, None] * stride_xn +
            ci[None, :] * stride_xc +
            ih * stride_xh +
            iw * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask_in, other=0.0)

        w_ptrs = w_ptr + (
            offs_n[None, :] * stride_wo +
            ci[:, None] * stride_wi +
            kh[:, None] * stride_wk +
            kw[:, None] * stride_wl
        )
        mask_w = (
            mask_k[:, None] & mask_n[None, :] &
            (ci[:, None] < C_in)
        )
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.dot(x, w, allow_tf32=True)

    # Add per-channel bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_conv2d_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                       stride=1, padding=0) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Triton conv2d expects CUDA tensors"
    assert x.dtype == weight.dtype == bias.dtype, "Input, weight, and bias must have same dtype"

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in_w == C_in, "Incompatible input / weight channels"
    assert bias.shape[0] == C_out, "Bias size must match out_channels"

    if isinstance(stride, int):
        stride_h_conv = stride
        stride_w_conv = stride
    else:
        stride_h_conv, stride_w_conv = stride

    if isinstance(padding, int):
        pad_h = padding
        pad_w = padding
    else:
        pad_h, pad_w = padding

    H_out = (H_in + 2 * pad_h - KH) // stride_h_conv + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w_conv + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wo, stride_wi, stride_wk, stride_wl = weight.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    M = N * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h_conv, stride_w_conv,
        pad_h, pad_w,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wi, stride_wk, stride_wl,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


@triton.jit
def fused_conv2_downsample_relu_kernel(
    act_ptr, x_ptr,
    w2_ptr, b2_ptr,
    wds_ptr, bds_ptr,
    y_ptr,
    N, C_mid, C_in,
    H0, W0,  # spatial size of x (residual input)
    H1, W1,  # spatial size of act (conv2 input) and output
    stride_ds,
    stride_an, stride_ac, stride_ah, stride_aw,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_w2o, stride_w2i, stride_w2k, stride_w2l,
    stride_wdso, stride_wdsi, stride_wdsk, stride_wdsl,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K2: tl.constexpr,
    BLOCK_KDS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = N * H1 * W1  # output elements over (N, H1, W1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_mid  # C_out == conv2.out_channels == C_mid in this block

    hw1 = H1 * W1
    n_idx = offs_m // hw1
    hw_idx = offs_m % hw1
    oh = hw_idx // W1
    ow = hw_idx % W1

    # Output pointers
    y_ptrs = y_ptr + (
        n_idx[:, None] * stride_yn +
        offs_n[None, :] * stride_yc +
        oh[:, None] * stride_yh +
        ow[:, None] * stride_yw
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- conv2 3x3 (on act_ptr) ----
    K2 = C_mid * 3 * 3
    for k_start in range(0, K2, BLOCK_K2):
        offs_k2 = k_start + tl.arange(0, BLOCK_K2)
        mask_k2 = offs_k2 < K2

        ci2 = offs_k2 // (3 * 3)
        rem2 = offs_k2 % (3 * 3)
        kh2 = rem2 // 3
        kw2 = rem2 % 3

        ih2 = oh[:, None] + kh2[None, :] - 1  # pad=1, stride=1
        iw2 = ow[:, None] + kw2[None, :] - 1

        mask_in2 = (
            mask_m[:, None] & mask_k2[None, :] &
            (ci2[None, :] < C_mid) &
            (ih2 >= 0) & (ih2 < H1) &
            (iw2 >= 0) & (iw2 < W1)
        )

        act_ptrs = act_ptr + (
            n_idx[:, None] * stride_an +
            ci2[None, :] * stride_ac +
            ih2 * stride_ah +
            iw2 * stride_aw
        )
        act = tl.load(act_ptrs, mask=mask_in2, other=0.0)

        w2_ptrs = w2_ptr + (
            offs_n[None, :] * stride_w2o +
            ci2[:, None] * stride_w2i +
            kh2[:, None] * stride_w2k +
            kw2[:, None] * stride_w2l
        )
        mask_w2 = (
            mask_k2[:, None] & mask_n[None, :] &
            (ci2[:, None] < C_mid)
        )
        w2 = tl.load(w2_ptrs, mask=mask_w2, other=0.0)

        acc += tl.dot(act, w2, allow_tf32=True)

    # Add conv2 bias
    b2 = tl.load(b2_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b2[None, :]

    # ---- downsample 1x1 conv (on x_ptr) ----
    # Output (oh, ow) corresponds to input (h0, w0) with stride_ds
    h0 = oh * stride_ds
    w0 = ow * stride_ds

    Kds = C_in  # 1x1, so reduction only over channels
    for kds_start in range(0, Kds, BLOCK_KDS):
        offs_kds = kds_start + tl.arange(0, BLOCK_KDS)
        mask_kds = offs_kds < Kds
        ci_ds = offs_kds

        mask_in_ds = (
            mask_m[:, None] & mask_kds[None, :] &
            (h0[:, None] >= 0) & (h0[:, None] < H0) &
            (w0[:, None] >= 0) & (w0[:, None] < W0)
        )

        x_ptrs = x_ptr + (
            n_idx[:, None] * stride_xn +
            ci_ds[None, :] * stride_xc +
            h0[:, None] * stride_xh +
            w0[:, None] * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask_in_ds, other=0.0)

        wds_ptrs = wds_ptr + (
            offs_n[None, :] * stride_wdso +
            ci_ds[:, None] * stride_wdsi
        )
        mask_wds = mask_kds[:, None] & mask_n[None, :]
        wds = tl.load(wds_ptrs, mask=mask_wds, other=0.0)

        acc += tl.dot(x, wds, allow_tf32=True)

    # Add downsample bias
    bds = tl.load(bds_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bds[None, :]

    # Final ReLU
    acc = tl.maximum(acc, 0.0)

    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def fused_conv2_downsample_relu(
    act: torch.Tensor,
    x: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    wds: torch.Tensor,
    bds: torch.Tensor,
    stride_ds: int,
) -> torch.Tensor:
    """
    Compute fused:
      out = ReLU( Conv2_BN(act) + Downsample_BN(x) )

    where Conv2_BN and Downsample_BN are represented by fused weights/biases.
    - act: output of conv1+BN1+ReLU, shape (N, C_mid, H1, W1)
    - x: original input, shape (N, C_in, H0, W0)
    """
    assert act.is_cuda and x.is_cuda, "Inputs must be CUDA tensors"
    assert w2.is_cuda and b2.is_cuda and wds.is_cuda and bds.is_cuda, "Weights/biases must be CUDA"
    assert act.dtype == x.dtype == w2.dtype == b2.dtype == wds.dtype == bds.dtype

    N, C_mid, H1, W1 = act.shape
    N2, C_in, H0, W0 = x.shape
    assert N == N2, "Batch size mismatch between act and x"

    C_out, C_mid_w, KH2, KW2 = w2.shape
    assert C_mid_w == C_mid and KH2 == 3 and KW2 == 3, "Unexpected conv2 weight shape"

    C_out2, C_in_w, KHds, KWds = wds.shape
    assert C_out2 == C_out and C_in_w == C_in and KHds == 1 and KWds == 1, "Unexpected downsample weight shape"

    stride_an, stride_ac, stride_ah, stride_aw = act.stride()
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_w2o, stride_w2i, stride_w2k, stride_w2l = w2.stride()
    stride_wdso, stride_wdsi, stride_wdsk, stride_wdsl = wds.stride()

    y = torch.empty((N, C_out, H1, W1), device=act.device, dtype=act.dtype)
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K2 = 32
    BLOCK_KDS = 32

    M = N * H1 * W1

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    fused_conv2_downsample_relu_kernel[grid](
        act, x,
        w2, b2,
        wds, bds,
        y,
        N, C_mid, C_in,
        H0, W0,
        H1, W1,
        stride_ds,
        stride_an, stride_ac, stride_ah, stride_aw,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_w2o, stride_w2i, stride_w2k, stride_w2l,
        stride_wdso, stride_wdsi, stride_wdsk, stride_wdsl,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K2=BLOCK_K2,
        BLOCK_KDS=BLOCK_KDS,
    )
    return y


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        # Fold BN into convs on-the-fly (using running stats) so that
        # runtime has only conv + pointwise ops.
        conv1_w, conv1_b = fold_conv_bn_2d(self.conv1, self.bn1)
        conv2_w, conv2_b = fold_conv_bn_2d(self.conv2, self.bn2)
        ds_conv, ds_bn = self.downsample[0], self.downsample[1]
        ds_w, ds_b = fold_conv_bn_2d(ds_conv, ds_bn)

        # conv1 + BN1 fused, then ReLU
        out = triton_conv2d_nchw(
            x,
            conv1_w,
            conv1_b,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
        )
        out = self.relu(out)

        # Fused conv2 (with BN2) + downsample (with BN) + residual add + ReLU
        out = fused_conv2_downsample_relu(
            out,
            x,
            conv2_w,
            conv2_b,
            ds_w,
            ds_b,
            stride_ds=self.stride,
        )
        return out
