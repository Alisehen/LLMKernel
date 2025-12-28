import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, y_ptr,
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

    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_conv2d_nchw(x: torch.Tensor, weight: torch.Tensor, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda, "Triton conv2d expects CUDA tensors"
    assert x.dtype == weight.dtype, "Input and weight must have same dtype"

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in_w == C_in, "Incompatible input / weight channels"

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
        x, weight, y,
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
def add_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    out = tl.maximum(out, 0.0)

    tl.store(c_ptr + offs, out, mask=mask)


def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "add_relu expects CUDA tensors"
    assert a.shape == b.shape, "Inputs must have the same shape"
    assert a.dtype == b.dtype, "Inputs must have the same dtype"

    N = a.numel()
    c = torch.empty_like(a)

    BLOCK = 256
    grid = lambda META: (triton.cdiv(N, META["BLOCK"]),)

    add_relu_kernel[grid](
        a.reshape(-1),
        b.reshape(-1),
        c.reshape(-1),
        N,
        BLOCK=BLOCK,
    )
    return c


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

    def _triton_conv(self, x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
        return triton_conv2d_nchw(
            x,
            conv.weight,
            stride=conv.stride,
            padding=conv.padding,
        )

    def forward(self, x):
        identity = x

        out = self._triton_conv(x, self.conv1)
        out = self.bn1(out)
        out = self.relu(out)

        out = self._triton_conv(out, self.conv2)
        out = self.bn2(out)

        if self.downsample is not None:
            ds_conv = self.downsample[0]
            ds_bn = self.downsample[1]
            identity = self._triton_conv(x, ds_conv)
            identity = ds_bn(identity)

        out = add_relu(out, identity)
        return out
