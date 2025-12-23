import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_im2col_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H_in, W_in,
    C_out, H_out, W_out,
    K,                           # K = C_in * KH * KW
    stride_h, stride_w,
    pad_h, pad_w,
    s_xn, s_xc, s_xh, s_xw,      # x strides
    s_wn, s_wc, s_wh, s_ww,      # w strides
    s_yn, s_yc, s_yh, s_yw,      # y strides
    M,                           # M = N * H_out * W_out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ADD_RELU: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row/col indices this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # decode (n, ho, wo) from flattened m index
    hw = offs_m % (H_out * W_out)
    ho = hw // W_out
    wo = hw % W_out
    n  = offs_m // (H_out * W_out)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K = C_in * KH * KW
    k0 = 0
    offs_k = tl.arange(0, BLOCK_K)
    while k0 < K:
        k_idx = k0 + offs_k
        mask_k = k_idx < K

        # map flat k index -> (c, kh, kw)
        c = k_idx // (KH * KW)
        khkw = k_idx % (KH * KW)
        kh = khkw // KW
        kw = khkw % KW

        # compute input coordinates for this (ho, wo, kh, kw)
        h_in = ho[:, None] * stride_h + kh[None, :] - pad_h
        w_in = wo[:, None] * stride_w + kw[None, :] - pad_w

        valid_h = (h_in >= 0) & (h_in < H_in)
        valid_w = (w_in >= 0) & (w_in < W_in)
        mask_x = (
            mask_m[:, None]
            & mask_k[None, :]
            & valid_h
            & valid_w
        )

        # input pointers: x[n, c, h_in, w_in]
        x_ptrs = (
            x_ptr
            + n[:, None] * s_xn
            + c[None, :] * s_xc
            + h_in * s_xh
            + w_in * s_xw
        )
        a = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # weight pointers: w[out_c, c, kh, kw]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * s_wn
            + c[:, None] * s_wc
            + kh[:, None] * s_wh
            + kw[:, None] * s_ww
        )
        mask_w = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # GEMM tile
        acc += tl.dot(a, b, allow_tf32=True)

        k0 += BLOCK_K

    # bias add
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    # ReLU
    if ADD_RELU:
        acc = tl.maximum(acc, 0.0)

    # write output y[n, out_c, ho, wo]
    y_ptrs = (
        y_ptr
        + n[:, None] * s_yn
        + offs_n[None, :] * s_yc
        + ho[:, None] * s_yh
        + wo[:, None] * s_yw
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def conv2d_triton(x, weight, bias=None, stride=1, padding=0, activation=None):
    """
    NCHW conv2d using Triton, implemented as an implicit-im2col GEMM.
    Supports:
      - groups=1, dilation=1
      - arbitrary batch size
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Incompatible in_channels"

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    # output spatial size (no dilation)
    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out),
                    device=x.device, dtype=torch.float32)

    # GEMM shapes: [M, K] @ [K, C_out]
    M = N * H_out * W_out
    K = C_in * KH * KW

    # tiling
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    has_bias = 1 if bias is not None else 0
    add_relu = 1 if activation == "relu" else 0
    b_ptr = bias if bias is not None else x  # dummy if unused

    conv2d_im2col_gemm_kernel[grid](
        x, weight, b_ptr, y,
        N, H_in, W_in,
        C_out, H_out, W_out,
        K,
        stride_h, stride_w,
        pad_h, pad_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KH=KH,
        KW=KW,
        HAS_BIAS=has_bias,
        ADD_RELU=add_relu,
    )

    return y


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
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

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    if ADD_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def linear_triton(x, weight, bias=None):
    """
    x: [M, K]
    weight: [N, K] (same as nn.Linear.weight)
    bias: [N] or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    # We compute x @ weight.T, but kernel expects B [K, N]
    b = weight.t().contiguous()  # [K, N]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    add_bias = 1 if bias is not None else 0
    bias_ptr = bias if bias is not None else x  # dummy

    linear_bias_kernel[grid](
        x, b, bias_ptr, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ADD_BIAS=add_bias,
    )

    return out


class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleNew, self).__init__()

        # 1x1 branch
        self.branch1x1_weight = nn.Parameter(torch.empty(out_1x1, in_channels, 1, 1))
        self.branch1x1_bias = nn.Parameter(torch.empty(out_1x1))

        # 3x3 branch
        self.branch3x3_reduce_weight = nn.Parameter(torch.empty(reduce_3x3, in_channels, 1, 1))
        self.branch3x3_reduce_bias = nn.Parameter(torch.empty(reduce_3x3))
        self.branch3x3_weight = nn.Parameter(torch.empty(out_3x3, reduce_3x3, 3, 3))
        self.branch3x3_bias = nn.Parameter(torch.empty(out_3x3))

        # 5x5 branch
        self.branch5x5_reduce_weight = nn.Parameter(torch.empty(reduce_5x5, in_channels, 1, 1))
        self.branch5x5_reduce_bias = nn.Parameter(torch.empty(reduce_5x5))
        self.branch5x5_weight = nn.Parameter(torch.empty(out_5x5, reduce_5x5, 5, 5))
        self.branch5x5_bias = nn.Parameter(torch.empty(out_5x5))

        # Pool branch
        self.branch_pool_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_proj_weight = nn.Parameter(torch.empty(pool_proj, in_channels, 1, 1))
        self.branch_pool_proj_bias = nn.Parameter(torch.empty(pool_proj))

        self._reset_parameters()

    def _reset_parameters(self):
        for w in [
            self.branch1x1_weight,
            self.branch3x3_reduce_weight,
            self.branch3x3_weight,
            self.branch5x5_reduce_weight,
            self.branch5x5_weight,
            self.branch_pool_proj_weight,
        ]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for b in [
            self.branch1x1_bias,
            self.branch3x3_reduce_bias,
            self.branch3x3_bias,
            self.branch5x5_reduce_bias,
            self.branch5x5_bias,
            self.branch_pool_proj_bias,
        ]:
            fan_in = b.numel()
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        # 1x1 branch
        branch1x1 = conv2d_triton(x, self.branch1x1_weight, self.branch1x1_bias, stride=1, padding=0)

        # 3x3 branch: 1x1 reduce -> 3x3 conv
        branch3x3 = conv2d_triton(x, self.branch3x3_reduce_weight, self.branch3x3_reduce_bias, stride=1, padding=0)
        branch3x3 = conv2d_triton(branch3x3, self.branch3x3_weight, self.branch3x3_bias, stride=1, padding=1)

        # 5x5 branch: 1x1 reduce -> 5x5 conv
        branch5x5 = conv2d_triton(x, self.branch5x5_reduce_weight, self.branch5x5_reduce_bias, stride=1, padding=0)
        branch5x5 = conv2d_triton(branch5x5, self.branch5x5_weight, self.branch5x5_bias, stride=1, padding=2)

        # Pool branch: maxpool -> 1x1 conv
        branch_pool = self.branch_pool_pool(x)
        branch_pool = conv2d_triton(branch_pool, self.branch_pool_proj_weight, self.branch_pool_proj_bias, stride=1, padding=0)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # conv1: 3 -> 64, 7x7, stride=2, padding=3
        self.conv1_weight = nn.Parameter(torch.empty(64, 3, 7, 7))
        self.conv1_bias = nn.Parameter(torch.empty(64))
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        # conv2: 64 -> 64, 1x1
        self.conv2_weight = nn.Parameter(torch.empty(64, 64, 1, 1))
        self.conv2_bias = nn.Parameter(torch.empty(64))

        # conv3: 64 -> 192, 3x3, padding=1
        self.conv3_weight = nn.Parameter(torch.empty(192, 64, 3, 3))
        self.conv3_bias = nn.Parameter(torch.empty(192))
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception modules
        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)

        # Fully connected: 1024 -> num_classes
        self.fc_weight = nn.Parameter(torch.empty(num_classes, 1024))
        self.fc_bias = nn.Parameter(torch.empty(num_classes))

        self._reset_parameters()

    def _reset_parameters(self):
        # Conv-like weights
        for w in [
            self.conv1_weight,
            self.conv2_weight,
            self.conv3_weight,
        ]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for b in [
            self.conv1_bias,
            self.conv2_bias,
            self.conv3_bias,
        ]:
            fan_in = b.numel()
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)

        # FC
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in = self.fc_weight.size(1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc_bias, -bound, bound)

    def forward(self, x):
        # x: [B, 3, H, W]

        # conv1 + ReLU + maxpool1
        x = conv2d_triton(x, self.conv1_weight, self.conv1_bias, stride=2, padding=3, activation="relu")
        x = self.maxpool1(x)

        # conv2 + ReLU
        x = conv2d_triton(x, self.conv2_weight, self.conv2_bias, stride=1, padding=0, activation="relu")

        # conv3 + ReLU + maxpool2
        x = conv2d_triton(x, self.conv3_weight, self.conv3_bias, stride=1, padding=1, activation="relu")
        x = self.maxpool2(x)

        # Inception 3a, 3b, maxpool3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4a-e, maxpool4
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5a, 5b
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Global avgpool
        x = self.avgpool(x)  # [B, 1024, 1, 1]
        x = torch.flatten(x, 1)  # [B, 1024]
        x = self.dropout(x)

        # Linear
        x = linear_triton(x, self.fc_weight, self.fc_bias)

        return x
