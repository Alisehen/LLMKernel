# Optimized Triton-based GoogLeNet-style model for RTX 4090
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------
# Triton Kernels
# -------------------------------

@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_HW: tl.constexpr,
    FUSE_RELU: tl.constexpr,
):
    """
    NCHW convolution, bias add, optional fused ReLU.
    Grid:
      pid_n  : batch dimension (0 .. N-1)
      pid_co : output-channel tile
      pid_hw : (H_out * W_out) tile
    All fused ops (bias, ReLU, store) share the same offsets and masks.
    """
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)  # [BLOCK_CO]
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    mask_co = offs_co < C_out
    mask_hw = offs_hw < H_out * W_out

    # Map linear hw index -> (h_out, w_out)
    offs_h = offs_hw // W_out
    offs_w = offs_hw - offs_h * W_out

    # Accumulator over [C_out_tile, HW_tile]
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # Loop over input channels and kernel spatial dims
    for ci in range(0, C_in):
        x_base_ci = x_ptr + pid_n * stride_xn + ci * stride_xc

        # Precompute input h indices and their bounds once per (ci, kh)
        for kh in tl.static_range(0, K_H):
            ih = offs_h * STRIDE_H + kh - PAD_H
            h_in_bounds = (ih >= 0) & (ih < H_in)

            for kw in tl.static_range(0, K_W):
                iw = offs_w * STRIDE_W + kw - PAD_W

                # Input bounds mask for this (kh, kw)
                in_bounds = (
                    mask_hw &
                    h_in_bounds &
                    (iw >= 0) & (iw < W_in)
                )

                # Input load (same hw offsets as output, masked by input bounds)
                x_ptrs = x_base_ci + ih * stride_xh + iw * stride_xw
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BLOCK_HW]

                # Weight load (same co offsets as output)
                w_ptrs = (
                    w_ptr
                    + offs_co * stride_wn
                    + ci * stride_wc
                    + kh * stride_wh
                    + kw * stride_ww
                )
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0)  # [BLOCK_CO]

                # FMA into accumulator
                acc += w_vals[:, None] * x_vals[None, :]

    # Bias add (fused, shares offsets/masks with store)
    b_vals = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)  # [BLOCK_CO]
    acc += b_vals[:, None]

    # Optional fused ReLU (same offsets/mask as bias/store)
    if FUSE_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store results
    y_base = y_ptr + pid_n * stride_yn
    y_ptrs = (
        y_base
        + offs_co[:, None] * stride_yc
        + offs_h[None, :] * stride_yh
        + offs_w[None, :] * stride_yw
    )
    out_mask = mask_co[:, None] & mask_hw[None, :]

    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.jit
def relu_inplace_kernel(
    x_ptr,
    N,
    BLOCK: tl.constexpr,
):
    """
    Simple 1D in-place ReLU.
    Grid: 1D over flattened tensor.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(x_ptr + offs, x, mask=mask)


@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Matrix multiplication C = A @ B + bias, where:
      A: [M, K]
      B: [K, N]
      bias: [N]
      C: [M, N]

    2D grid over (M, N). Fused bias add shares offsets/masks with store.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add (shares offs_n / mask with store)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BN]
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# -------------------------------
# Triton Wrapper Functions
# -------------------------------

def conv2d_triton(x, weight, bias, stride=1, padding=0, fuse_relu=False):
    """
    NCHW convolution wrapper using conv2d_nchw_kernel.
    Supports stride/padding (int or tuple). 'fuse_relu' toggles fused ReLU.
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv2d requires CUDA tensors"
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Inconsistent in_channels between input and weight"

    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h = stride_w = stride
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    H_out = (H_in + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K_w) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Tiling: 64 x 64 is a good default for RTX 4090
    BLOCK_CO = 64
    BLOCK_HW = 64

    grid = (
        N,
        triton.cdiv(C_out, BLOCK_CO),
        triton.cdiv(H_out * W_out, BLOCK_HW),
    )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        STRIDE_H=stride_h,
        STRIDE_W=stride_w,
        PAD_H=pad_h,
        PAD_W=pad_w,
        K_H=K_h,
        K_W=K_w,
        BLOCK_CO=BLOCK_CO,
        BLOCK_HW=BLOCK_HW,
        FUSE_RELU=fuse_relu,
        num_warps=4,
        num_stages=2,
    )

    return y


def relu_inplace_triton(x):
    """
    In-place ReLU on an arbitrary CUDA tensor.
    """
    assert x.is_cuda, "Triton ReLU requires CUDA tensor"
    N = x.numel()
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    relu_inplace_kernel[grid](x, N, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return x


def linear_triton(x, weight, bias):
    """
    Linear layer: y = x @ weight.T + bias
    Uses fused matmul + bias-add kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Triton Linear requires CUDA tensors"
    M, K = x.shape
    out_features, in_features = weight.shape
    assert K == in_features, "Inconsistent in_features"

    # Transform weight to (K, N) for GEMM
    B = weight.t().contiguous()
    N = out_features

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    linear_kernel[grid](
        x, B, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return y


# -------------------------------
# Triton-Based Modules
# -------------------------------

class TritonConv2d(nn.Module):
    """
    Pure convolution (no activation), used where no ReLU is applied afterwards.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, tuple):
            k_h, k_w = kernel_size
        else:
            k_h = k_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k_h, k_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Kaiming init similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * k_h * k_w
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias = self.bias
        if bias is None:
            # use zero bias tensor on same device/dtype
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        return conv2d_triton(
            x, self.weight, bias,
            stride=self.stride,
            padding=self.padding,
            fuse_relu=False,
        )


class TritonConv2dReLU(nn.Module):
    """
    Convolution fused with ReLU, used where a ReLU follows immediately.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, tuple):
            k_h, k_w = kernel_size
        else:
            k_h = k_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k_h, k_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Kaiming init similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_channels * k_h * k_w
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias = self.bias
        if bias is None:
            bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
        # Fused conv + bias + ReLU, single grid / offsets / mask
        return conv2d_triton(
            x, self.weight, bias,
            stride=self.stride,
            padding=self.padding,
            fuse_relu=True,
        )


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        # Kaiming-like init similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        bias = self.bias
        if bias is None:
            bias = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        return linear_triton(x, self.weight, bias)


# -------------------------------
# Inception Module (Triton-based)
# -------------------------------

class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1,
                 reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5,
                 pool_proj):
        super().__init__()

        # 1x1 convolution branch (no activation here, matches original ModelNew)
        self.branch1x1 = TritonConv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3_1 = TritonConv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = TritonConv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        # 5x5 convolution branch
        self.branch5x5_1 = TritonConv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5_2 = TritonConv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        # Max pooling branch
        self.branch_pool_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_conv = TritonConv2d(in_channels, pool_proj, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = self.branch_pool_pool(x)
        branch_pool = self.branch_pool_conv(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)


# -------------------------------
# Top-Level Model (Triton-based)
# -------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Use fused Conv+ReLU where Model originally had explicit ReLU after conv
        self.conv1 = TritonConv2dReLU(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = TritonConv2dReLU(64, 64, kernel_size=1)
        self.conv3 = TritonConv2dReLU(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

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
        self.fc = TritonLinear(1024, num_classes)

    def forward(self, x):
        # x: (batch_size, 3, H, W)
        x = self.conv1(x)        # fused conv+ReLU
        x = self.maxpool1(x)

        x = self.conv2(x)        # fused conv+ReLU
        x = self.conv3(x)        # fused conv+ReLU
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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
