import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================================================
#   Helper "missing" Triton math functions (scalar elementwise)
# ============================================================

@triton.jit
def tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def tl_tanh(x):
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    exp2x = tl.exp(2.0 * x)
    return (exp2x - 1.0) / (exp2x + 1.0)


@triton.jit
def tl_gelu(x):
    # Approx GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def tl_silu(x):
    # x * sigmoid(x)
    return x * tl_sigmoid(x)


@triton.jit
def tl_softmax(x, axis: tl.constexpr):
    # Simple numerically stable softmax along given axis
    x_max = tl.max(x, axis=axis)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    denom = tl.sum(exp_x, axis=axis)
    return exp_x / denom


@triton.jit
def tl_mish(x):
    # mish(x) = x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    return x * tl_tanh(softplus)


# ============================================================
#  Triton Conv2d (NCHW) : im2col + GEMM + bias + optional ReLU
#  Autotuned BLOCK sizes with register-pressure-aware tiles
# ============================================================

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["out_elems", "C_out", "K"],
)
@triton.jit
def conv2d_im2col_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    K,                # = C_in * KH * KW
    out_elems,        # = N * H_out * W_out
    BLOCK_M: tl.constexpr,   # tile in (N * H_out * W_out)
    BLOCK_N: tl.constexpr,   # tile in C_out
    BLOCK_K: tl.constexpr,   # tile in K (reduction)
    KH: tl.constexpr,
    KW: tl.constexpr,
    APPLY_RELU: tl.constexpr,
):
    # 2D program grid over output tensor:
    # dim0: flattened (N * H_out * W_out)   -> M dimension
    # dim1: C_out                            -> N dimension
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BM,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BN,)

    # Masks for output tile
    mask_m = offs_m < out_elems
    mask_n = offs_n < C_out
    out_mask = mask_m[:, None] & mask_n[None, :]

    HW_out = H_out * W_out
    KW_total = KH * KW

    offs_m_b = offs_m[:, None]  # (BM,1) for broadcasting
    n_idx = offs_m_b // HW_out
    tmp = offs_m_b % HW_out
    ho = tmp // W_out
    wo = tmp % W_out

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = C_in * KH * KW
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # (BK,)
        mask_k = offs_k < K

        offs_k_b = offs_k[None, :]  # (1,BK)

        # Decode K index -> (ci, kh, kw)
        ci = offs_k_b // KW_total
        tmp2 = offs_k_b % KW_total
        kh = tmp2 // KW
        kw = tmp2 % KW

        # Input spatial positions
        h_in = ho * stride_h + kh - pad_h  # (BM,BK)
        w_in = wo * stride_w + kw - pad_w  # (BM,BK)

        a_mask = (
            mask_m[:, None]
            & mask_k[None, :]
            & (h_in >= 0)
            & (h_in < H_in)
            & (w_in >= 0)
            & (w_in < W_in)
        )

        # Input index: ((n*C_in + ci)*H_in + h_in)*W_in + w_in
        idx_in = (((n_idx * C_in) + ci) * H_in + h_in) * W_in + w_in
        a = tl.load(x_ptr + idx_in, mask=a_mask, other=0.0)

        # Weight matrix w_ptr: shape [K, C_out] row-major
        b_ptrs = w_ptr + offs_k[:, None] * C_out + offs_n[None, :]
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    # Fused bias add (broadcast over BM)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store: y has shape [N, C_out, H_out, W_out]
    out_idx = (((n_idx * C_out) + offs_n[None, :]) * H_out + ho) * W_out + wo
    tl.store(y_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d_nchw(x, weight, bias, stride=1, padding=0, apply_relu=False):
    """
    x:      (N, C_in, H_in, W_in), float32, contiguous
    weight: (C_out, C_in, KH, KW), contiguous
    bias:   (C_out,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_w, KH, KW = weight.shape
    assert C_w == C_in

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = int(stride)

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = int(padding)

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    Kdim = C_in * KH * KW
    out_elems = N * H_out * W_out

    # Weight matrix [K, C_out]
    w_2d = weight.view(C_out, -1).transpose(0, 1).contiguous()

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(out_elems, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_im2col_gemm_kernel[grid](
        x, w_2d, bias, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        Kdim,
        out_elems,
        KH=KH,
        KW=KW,
        APPLY_RELU=apply_relu,
    )
    return y


# ============================================================
#  Triton MaxPool2d (NCHW)
# ============================================================

@triton.jit
def maxpool2d_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    # 1D grid over all output elements (N * C * H_out * W_out)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total_out = N * C * H_out * W_out
    mask = offs < total_out

    # Decode (n, c, ho, wo)
    hw_out = H_out * W_out
    nc_hw = offs // hw_out
    ho = (offs % hw_out) // W_out
    wo = offs % W_out
    n = nc_hw // C
    c = nc_hw % C

    hi0 = ho * stride_h - pad_h
    wi0 = wo * stride_w - pad_w

    # Initialize to minimum float32
    min_val = -3.4028235e38
    max_val = tl.full((BLOCK,), min_val, dtype=tl.float32)

    for kh in range(KH):
        hi = hi0 + kh
        for kw in range(KW):
            wi = wi0 + kw
            load_mask = (
                mask
                & (hi >= 0)
                & (hi < H_in)
                & (wi >= 0)
                & (wi < W_in)
            )
            idx_in = ((n * C + c) * H_in + hi) * W_in + wi
            val = tl.load(x_ptr + idx_in, mask=load_mask, other=min_val)
            max_val = tl.maximum(max_val, val)

    out_idx = ((n * C + c) * H_out + ho) * W_out + wo
    tl.store(y_ptr + out_idx, max_val, mask=mask)


def triton_maxpool2d_nchw(x, kernel_size=3, stride=2, padding=0):
    """
    x: (N, C, H_in, W_in), contiguous
    """
    assert x.is_cuda
    x = x.contiguous()
    N, C, H_in, W_in = x.shape

    if isinstance(kernel_size, (tuple, list)):
        KH, KW = kernel_size
    else:
        KH = KW = int(kernel_size)

    if isinstance(stride, (tuple, list)):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = int(stride)

    if isinstance(padding, (tuple, list)):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = int(padding)

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    total_out = N * C * H_out * W_out

    def grid(meta):
        return (triton.cdiv(total_out, meta["BLOCK"]),)

    maxpool2d_kernel[grid](
        x, y,
        N, C, H_in, W_in,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK=256,
        KH=KH,
        KW=KW,
        num_warps=4,
        num_stages=1,
    )
    return y


# ============================================================
#  Triton AdaptiveAvgPool2d to (1,1)
# ============================================================

@triton.jit
def adaptive_avgpool2d_1x1_kernel(
    x_ptr, y_ptr,
    NC, S,                # S = H * W
    BLOCK_NC: tl.constexpr,
):
    # 1D grid over NC = N * C
    pid = tl.program_id(0)
    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    mask_nc = offs_nc < NC

    base = offs_nc * S

    acc = tl.zeros((BLOCK_NC,), dtype=tl.float32)

    for s in range(0, S):
        vals = tl.load(x_ptr + base + s, mask=mask_nc, other=0.0)
        acc += vals

    avg = acc / S
    tl.store(y_ptr + offs_nc, avg, mask=mask_nc)


def triton_adaptive_avgpool2d_1x1(x):
    """
    x: (N, C, H, W) -> (N, C, 1, 1)
    """
    assert x.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    S = H * W
    NC = N * C

    y_flat = torch.empty(NC, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(NC, meta["BLOCK_NC"]),)

    adaptive_avgpool2d_1x1_kernel[grid](
        x, y_flat,
        NC, S,
        BLOCK_NC=256,
        num_warps=4,
        num_stages=1,
    )

    y = y_flat.view(N, C, 1, 1)
    return y


# ============================================================
#  Triton Linear: GEMM + Bias + optional ReLU (M x K) * (K x N)
#  Autotuned tiles, register-pressure-aware
# ============================================================

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Base pointers for this tile (start at k=0)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_index = k0 + offs_k
        mask_k = k_index < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add (broadcast along M)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=out_mask,
    )


def triton_linear(x, weight, bias, apply_relu=False):
    """
    x: (M, K)
    weight: (N, K)  (same as nn.Linear.out_features, in_features)
    bias: (N,)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K_w == K

    # b: (K, N) for GEMM
    b = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_gemm_bias_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        APPLY_RELU=apply_relu,
    )
    return c


# ============================================================
#  Small Triton-based Layers
# ============================================================

class TritonConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            assert kernel_size[0] == kernel_size[1]
            k = kernel_size[0]
        else:
            k = int(kernel_size)
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, apply_relu: bool = False):
        return triton_conv2d_nchw(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            apply_relu=apply_relu,
        )


class TritonMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            assert kernel_size[0] == kernel_size[1]
            k = kernel_size[0]
        else:
            k = int(kernel_size)
        self.kernel_size = k
        self.stride = stride if stride is not None else k
        self.padding = padding

    def forward(self, x):
        return triton_maxpool2d_nchw(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )


class TritonAdaptiveAvgPool2d1x1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_adaptive_avgpool2d_1x1(x)


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, apply_relu: bool = False):
        return triton_linear(x, self.weight, self.bias, apply_relu=apply_relu)


class IdentityDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Original dropout has p=0.0, so this is an exact replacement.
        return x


# ============================================================
#  Inception Module (Triton)
# ============================================================

class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1,
                 reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5,
                 pool_proj):
        super(InceptionModuleNew, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = TritonConv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3_1 = TritonConv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = TritonConv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        # 5x5 convolution branch
        self.branch5x5_1 = TritonConv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = TritonConv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        # Max pooling branch
        self.branch_pool_pool = TritonMaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = TritonConv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        # Note: Original module has no explicit ReLU in branches, so we keep that.
        branch1x1 = self.branch1x1(x)

        b3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(b3)

        b5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(b5)

        pool = self.branch_pool_pool(x)
        branch_pool = self.branch_pool_conv(pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)


# ============================================================
#  Full Model using Triton kernels
# ============================================================

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = TritonConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = TritonMaxPool2d(3, stride=2, padding=1)
        self.conv2 = TritonConv2d(64, 64, kernel_size=1)
        self.conv3 = TritonConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = TritonMaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = TritonMaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = TritonMaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = TritonAdaptiveAvgPool2d1x1()
        self.dropout = IdentityDropout()
        self.fc = TritonLinear(1024, num_classes)

    def forward(self, x):
        # x: (batch_size, 3, H, W)
        # conv1 + ReLU + maxpool1
        x = self.conv1(x, apply_relu=True)
        x = self.maxpool1(x)

        # conv2 + ReLU
        x = self.conv2(x, apply_relu=True)

        # conv3 + ReLU + maxpool2
        x = self.conv3(x, apply_relu=True)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)         # (N, 1024, 1, 1)
        x = torch.flatten(x, 1)     # (N, 1024)
        x = self.dropout(x)         # identity (p=0)
        x = self.fc(x)              # (N, num_classes)
        return x
