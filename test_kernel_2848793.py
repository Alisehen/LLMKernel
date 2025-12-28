import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_bn_relu_kernel(
    x_ptr, w_ptr,
    bn_weight_ptr, bn_bias_ptr, bn_mean_ptr, bn_var_ptr,
    y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_total,
    K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wo,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    RELU: tl.constexpr, APPLY_BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_total = N * H_out * W_out

    mask_m = offs_m < M_total
    mask_o = offs_o < C_out

    # Decode output indices from offs_m
    hw_out = H_out * W_out
    n = offs_m // hw_out
    rem = offs_m % hw_out
    oh = rem // W_out
    ow = rem % W_out

    # Precompute for reduction mapping
    KW_total = K_h * K_w

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K_total:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        # Map reduction index to (c_in, kh, kw)
        c_in = offs_k // KW_total
        remk = offs_k % KW_total
        kh = remk // K_w
        kw = remk % K_w

        # Compute input coordinates
        ih = oh[:, None] * stride_h + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] - pad_w

        in_bounds_h = (ih >= 0) & (ih < H_in)
        in_bounds_w = (iw >= 0) & (iw < W_in)
        mask_in = in_bounds_h & in_bounds_w

        # Build pointers for X
        a_ptrs = (
            x_ptr
            + n[:, None] * stride_xn
            + c_in[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )

        mask_a = mask_m[:, None] & mask_k[None, :] & mask_in
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Build pointers for W: [K_total, C_out]
        b_ptrs = (
            w_ptr
            + offs_k[:, None] * stride_wk
            + offs_o[None, :] * stride_wo
        )
        mask_b = mask_k[:, None] & mask_o[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        k0 += BLOCK_K

    # BatchNorm (inference) + ReLU
    if APPLY_BN:
        gamma = tl.load(bn_weight_ptr + offs_o, mask=mask_o, other=1.0)
        beta = tl.load(bn_bias_ptr + offs_o, mask=mask_o, other=0.0)
        mean = tl.load(bn_mean_ptr + offs_o, mask=mask_o, other=0.0)
        var = tl.load(bn_var_ptr + offs_o, mask=mask_o, other=1.0)
        inv_std = 1.0 / tl.sqrt(var + eps)
        gamma = gamma[None, :]
        beta = beta[None, :]
        mean = mean[None, :]
        inv_std = inv_std[None, :]
        acc = (acc - mean) * inv_std * gamma + beta

    if RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_yn
        + offs_o[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    mask_y = mask_m[:, None] & mask_o[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)


def triton_conv2d_bn_relu(
    x,
    weight,
    bn_weight,
    bn_bias,
    bn_mean,
    bn_var,
    stride=1,
    padding=0,
    eps=1e-5,
    relu=True,
):
    x = x.contiguous()
    weight = weight.contiguous()
    bn_weight = bn_weight.contiguous()
    bn_bias = bn_bias.contiguous()
    bn_mean = bn_mean.contiguous()
    bn_var = bn_var.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_w_in, K_h, K_w = weight.shape
    assert C_in == C_w_in

    H_out = (H_in + 2 * padding - K_h) // stride + 1
    W_out = (W_in + 2 * padding - K_w) // stride + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    K_total = C_in * K_h * K_w
    w_col = weight.view(C_out, K_total).transpose(0, 1).contiguous()

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_bn_relu_kernel[grid](
        x, w_col,
        bn_weight, bn_bias, bn_mean, bn_var,
        y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_total,
        K_h, K_w,
        stride, stride,
        padding, padding,
        eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_col.stride(0), w_col.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        RELU=1 if relu else 0,
        APPLY_BN=1,
    )
    return y


@triton.jit
def maxpool2d_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    M = N * C * H_out * W_out
    mask = offs < M

    tmp = offs // (H_out * W_out)
    n = tmp // C
    c = tmp % C
    rem = offs % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    # Initialize accumulator to very small value
    acc = tl.full((BLOCK,), -1e30, dtype=tl.float32)

    kh = 0
    while kh < kernel_h:
        kw = 0
        while kw < kernel_w:
            ih = oh * stride_h - pad_h + kh
            iw = ow * stride_w - pad_w + kw

            in_bounds = (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in) & mask

            x_ptrs = (
                x_ptr
                + n * stride_xn
                + c * stride_xc
                + ih * stride_xh
                + iw * stride_xw
            )
            vals = tl.load(x_ptrs, mask=in_bounds, other=-1e30)
            acc = tl.maximum(acc, vals)
            kw += 1
        kh += 1

    y_ptrs = (
        y_ptr
        + n * stride_yn
        + c * stride_yc
        + oh * stride_yh
        + ow * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask)


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    x = x.contiguous()
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(N * C * H_out * W_out, META["BLOCK"]),)

    maxpool2d_kernel[grid](
        x, y,
        N, C, H_in, W_in,
        H_out, W_out,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=128,
    )
    return y


@triton.jit
def adaptive_avgpool2d_1x1_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
):
    pid = tl.program_id(0)
    # Each program handles one (n, c)
    nc = pid
    n = nc // C
    c = nc % C

    acc = 0.0
    h = 0
    while h < H:
        w = 0
        while w < W:
            offset = n * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw
            val = tl.load(x_ptr + offset)
            acc += val
            w += 1
        h += 1

    factor = 1.0 / (H * W)
    acc = acc * factor
    out_offset = n * stride_yn + c * stride_yc
    tl.store(y_ptr + out_offset, acc)


def triton_adaptive_avgpool2d_1x1(x):
    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(N * C, 1),)

    adaptive_avgpool2d_1x1_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1),
    )
    return y


@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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

    k0 = 0
    while k0 < K:
        k_mask = offs_k[None, :] + k0 < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask.T & mask_n[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k0 += BLOCK_K

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x, weight, bias=None):
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]  # weight: [out_features, in_features]
    w_t = weight.t().contiguous()  # [K, N]

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
        x, w_t, bias if bias is not None else x, c,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=1 if bias is not None else 0,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c


class BasicBlockTriton(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, eps=1e-5):
        super(BasicBlockTriton, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.eps = eps

        # conv1: 3x3
        self.conv1_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, 3, 3) * (2.0 / (in_channels * 3 * 3)) ** 0.5
        )
        self.bn1_weight = nn.Parameter(torch.ones(out_channels))
        self.bn1_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn1_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.bn1_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)

        # conv2: 3x3
        self.conv2_weight = nn.Parameter(
            torch.randn(out_channels, out_channels, 3, 3) * (2.0 / (out_channels * 3 * 3)) ** 0.5
        )
        self.bn2_weight = nn.Parameter(torch.ones(out_channels))
        self.bn2_bias = nn.Parameter(torch.zeros(out_channels))
        self.bn2_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
        self.bn2_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)

        self.has_downsample = downsample
        if self.has_downsample:
            # 1x1 downsample conv
            self.down_conv_weight = nn.Parameter(
                torch.randn(out_channels, in_channels, 1, 1) * (2.0 / (in_channels)) ** 0.5
            )
            self.down_bn_weight = nn.Parameter(torch.ones(out_channels))
            self.down_bn_bias = nn.Parameter(torch.zeros(out_channels))
            self.down_bn_running_mean = nn.Parameter(torch.zeros(out_channels), requires_grad=False)
            self.down_bn_running_var = nn.Parameter(torch.ones(out_channels), requires_grad=False)

    def forward(self, x):
        identity = x

        out = triton_conv2d_bn_relu(
            x,
            self.conv1_weight,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_running_mean,
            self.bn1_running_var,
            stride=self.stride,
            padding=1,
            eps=self.eps,
            relu=True,
        )

        out = triton_conv2d_bn_relu(
            out,
            self.conv2_weight,
            self.bn2_weight,
            self.bn2_bias,
            self.bn2_running_mean,
            self.bn2_running_var,
            stride=1,
            padding=1,
            eps=self.eps,
            relu=False,
        )

        if self.has_downsample:
            identity = triton_conv2d_bn_relu(
                x,
                self.down_conv_weight,
                self.down_bn_weight,
                self.down_bn_bias,
                self.down_bn_running_mean,
                self.down_bn_running_var,
                stride=self.stride,
                padding=0,
                eps=self.eps,
                relu=False,
            )

        out = out + identity
        out = torch.relu(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64
        self.eps = 1e-5

        # Initial conv1: 7x7, stride 2, padding 3
        self.conv1_weight = nn.Parameter(
            torch.randn(64, 3, 7, 7) * (2.0 / (3 * 7 * 7)) ** 0.5
        )
        self.bn1_weight = nn.Parameter(torch.ones(64))
        self.bn1_bias = nn.Parameter(torch.zeros(64))
        self.bn1_running_mean = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.bn1_running_var = nn.Parameter(torch.ones(64), requires_grad=False)

        # Layers
        self.layer1 = self._make_layer(BasicBlockTriton, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockTriton, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockTriton, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockTriton, 512, 2, stride=2)

        # FC
        self.fc_weight = nn.Parameter(
            torch.randn(num_classes, 512 * BasicBlockTriton.expansion)
            * (2.0 / (512 * BasicBlockTriton.expansion)) ** 0.5
        )
        self.fc_bias = nn.Parameter(torch.zeros(num_classes))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = stride != 1 or self.in_channels != out_channels * block.expansion

        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample, eps=self.eps))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, downsample=False, eps=self.eps))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, H, W]
        x = triton_conv2d_bn_relu(
            x,
            self.conv1_weight,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_running_mean,
            self.bn1_running_var,
            stride=2,
            padding=3,
            eps=self.eps,
            relu=True,
        )

        x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = triton_adaptive_avgpool2d_1x1(x)
        x = torch.flatten(x, 1)
        x = triton_linear(x, self.fc_weight, self.fc_bias)

        return x
