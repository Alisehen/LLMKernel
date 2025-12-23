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
    C_out, K_h, K_w,
    stride_h, stride_w,
    pad_h, pad_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_co = offs_co < C_out
    mask_hw = offs_hw < H_out * W_out

    # Map linear hw index -> (h_out, w_out)
    offs_h = offs_hw // W_out
    offs_w = offs_hw - offs_h * W_out

    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # Loop over input channels and kernel spatial dims
    for ci in range(0, C_in):
        for kh in range(0, K_h):
            for kw in range(0, K_w):
                ih = offs_h * stride_h + kh - pad_h
                iw = offs_w * stride_w + kw - pad_w

                in_bounds = (
                    mask_hw &
                    (ih >= 0) & (ih < H_in) &
                    (iw >= 0) & (iw < W_in)
                )

                x_base = x_ptr + pid_n * stride_xn + ci * stride_xc
                x_ptrs = x_base + ih * stride_xh + iw * stride_xw
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

                w_base = w_ptr + offs_co * stride_wn + ci * stride_wc + kh * stride_wh + kw * stride_ww
                w_vals = tl.load(w_base, mask=mask_co, other=0.0)

                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)
    acc += b_vals[:, None]

    # Store results
    y_base = y_ptr + pid_n * stride_yn
    y_ptrs = y_base + offs_co[:, None] * stride_yc + offs_h[None, :] * stride_yh + offs_w[None, :] * stride_yw
    out_mask = mask_co[:, None] & mask_hw[None, :]

    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.jit
def relu_inplace_kernel(
    x_ptr,
    N,
    BLOCK: tl.constexpr,
):
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# -------------------------------
# Triton Wrapper Functions
# -------------------------------

def conv2d_triton(x, weight, bias, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda, "Triton conv2d requires CUDA tensors"
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Inconsistent in_channels"

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
        C_out, K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
    )

    return y


def relu_inplace_triton(x):
    assert x.is_cuda, "Triton ReLU requires CUDA tensor"
    N = x.numel()
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    relu_inplace_kernel[grid](x, N, BLOCK=BLOCK)
    return x


def linear_triton(x, weight, bias):
    assert x.is_cuda and weight.is_cuda, "Triton Linear requires CUDA tensors"
    M, K = x.shape
    out_features, in_features = weight.shape
    assert K == in_features, "Inconsistent in_features"

    # Weight for GEMM: (K, N)
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
    )
    return y


# -------------------------------
# Triton-Based Modules
# -------------------------------

class TritonConv2d(nn.Module):
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

        # 1x1 convolution branch
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

        self.conv1 = TritonConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = TritonConv2d(64, 64, kernel_size=1)
        self.conv3 = TritonConv2d(64, 192, kernel_size=3, padding=1)
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
        x = self.conv1(x)
        x = relu_inplace_triton(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = relu_inplace_triton(x)

        x = self.conv3(x)
        x = relu_inplace_triton(x)
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
