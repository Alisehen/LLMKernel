import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def fused_gemm_bias_kernel(
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C, H, W, H_out, W_out, STRIDE,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    nc = pid_nc
    n = nc // C
    c = nc % C

    h_out = pid_h
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_out = (n < N) & (c < C) & (h_out < H_out) & (offs_w < W_out)

    # Early exit is not allowed inside kernels; rely on masking instead.

    # Compute input coordinate starts (with padding=1)
    h_in_start = h_out * STRIDE - 1
    w_in_start = offs_w * STRIDE - 1

    # Load 3x3 weights for this channel (C,1,3,3) -> 9 scalars
    w_base = w_ptr + c * 9
    w00 = tl.load(w_base + 0)
    w01 = tl.load(w_base + 1)
    w02 = tl.load(w_base + 2)
    w10 = tl.load(w_base + 3)
    w11 = tl.load(w_base + 4)
    w12 = tl.load(w_base + 5)
    w20 = tl.load(w_base + 6)
    w21 = tl.load(w_base + 7)
    w22 = tl.load(w_base + 8)

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Precompute h positions (scalars)
    h0 = h_in_start + 0
    h1 = h_in_start + 1
    h2 = h_in_start + 2

    # Precompute w positions (vectors)
    w0 = w_in_start + 0
    w1 = w_in_start + 1
    w2 = w_in_start + 2

    # Helper to load and accumulate for each (hi, wj, wij)
    # (0,0)
    mask_00 = mask_out & (h0 >= 0) & (h0 < H) & (w0 >= 0) & (w0 < W)
    x_ptrs_00 = x_ptr + n * stride_xn + c * stride_xc + h0 * stride_xh + w0 * stride_xw
    x00 = tl.load(x_ptrs_00, mask=mask_00, other=0.0)
    acc += x00 * w00

    # (0,1)
    mask_01 = mask_out & (h0 >= 0) & (h0 < H) & (w1 >= 0) & (w1 < W)
    x_ptrs_01 = x_ptr + n * stride_xn + c * stride_xc + h0 * stride_xh + w1 * stride_xw
    x01 = tl.load(x_ptrs_01, mask=mask_01, other=0.0)
    acc += x01 * w01

    # (0,2)
    mask_02 = mask_out & (h0 >= 0) & (h0 < H) & (w2 >= 0) & (w2 < W)
    x_ptrs_02 = x_ptr + n * stride_xn + c * stride_xc + h0 * stride_xh + w2 * stride_xw
    x02 = tl.load(x_ptrs_02, mask=mask_02, other=0.0)
    acc += x02 * w02

    # (1,0)
    mask_10 = mask_out & (h1 >= 0) & (h1 < H) & (w0 >= 0) & (w0 < W)
    x_ptrs_10 = x_ptr + n * stride_xn + c * stride_xc + h1 * stride_xh + w0 * stride_xw
    x10 = tl.load(x_ptrs_10, mask=mask_10, other=0.0)
    acc += x10 * w10

    # (1,1)
    mask_11 = mask_out & (h1 >= 0) & (h1 < H) & (w1 >= 0) & (w1 < W)
    x_ptrs_11 = x_ptr + n * stride_xn + c * stride_xc + h1 * stride_xh + w1 * stride_xw
    x11 = tl.load(x_ptrs_11, mask=mask_11, other=0.0)
    acc += x11 * w11

    # (1,2)
    mask_12 = mask_out & (h1 >= 0) & (h1 < H) & (w2 >= 0) & (w2 < W)
    x_ptrs_12 = x_ptr + n * stride_xn + c * stride_xc + h1 * stride_xh + w2 * stride_xw
    x12 = tl.load(x_ptrs_12, mask=mask_12, other=0.0)
    acc += x12 * w12

    # (2,0)
    mask_20 = mask_out & (h2 >= 0) & (h2 < H) & (w0 >= 0) & (w0 < W)
    x_ptrs_20 = x_ptr + n * stride_xn + c * stride_xc + h2 * stride_xh + w0 * stride_xw
    x20 = tl.load(x_ptrs_20, mask=mask_20, other=0.0)
    acc += x20 * w20

    # (2,1)
    mask_21 = mask_out & (h2 >= 0) & (h2 < H) & (w1 >= 0) & (w1 < W)
    x_ptrs_21 = x_ptr + n * stride_xn + c * stride_xc + h2 * stride_xh + w1 * stride_xw
    x21 = tl.load(x_ptrs_21, mask=mask_21, other=0.0)
    acc += x21 * w21

    # (2,2)
    mask_22 = mask_out & (h2 >= 0) & (h2 < H) & (w2 >= 0) & (w2 < W)
    x_ptrs_22 = x_ptr + n * stride_xn + c * stride_xc + h2 * stride_xh + w2 * stride_xw
    x22 = tl.load(x_ptrs_22, mask=mask_22, other=0.0)
    acc += x22 * w22

    y_ptrs = y_ptr + n * stride_yn + c * stride_yc + h_out * stride_yh + offs_w * stride_yw
    tl.store(y_ptrs, acc, mask=mask_out)


def conv1x1_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    1x1 convolution implemented as GEMM:
    x: [N, C_in, H, W]
    weight: [C_out, C_in, 1, 1]
    """
    assert x.is_cuda and weight.is_cuda
    assert weight.dim() == 4 and weight.shape[2] == 1 and weight.shape[3] == 1

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]

    # Reshape to [M, K] @ [K, N] pattern
    x_2d = x.permute(0, 2, 3, 1).reshape(-1, C_in).contiguous()  # [M, K]
    w_2d = weight.reshape(C_out, C_in).contiguous()              # [C_out, C_in]
    B = w_2d.t().contiguous()                                    # [K, C_out]

    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N_out = B.shape[1]

    y_2d = torch.empty((M, N_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N_out, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        x_2d, B, y_2d,
        M, N_out, K,
        x_2d.stride(0), x_2d.stride(1),
        B.stride(0), B.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    y = y_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    return y


def depthwise_conv3x3_triton(x: torch.Tensor, weight: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Depthwise 3x3 convolution with padding=1, arbitrary stride (1 or 2).
    x: [N, C, H, W]
    weight: [C, 1, 3, 3] (groups = C)
    """
    assert x.is_cuda and weight.is_cuda
    N, C, H, W = x.shape
    assert weight.shape[0] == C and weight.shape[1] == 1 and weight.shape[2] == 3 and weight.shape[3] == 3

    pad = 1
    H_out = (H + 2 * pad - 3) // stride + 1
    W_out = (W + 2 * pad - 3) // stride + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        N * C,
        H_out,
        triton.cdiv(W_out, META['BLOCK_W']),
    )

    depthwise_conv3x3_kernel[grid](
        x, weight, y,
        N, C, H, W, H_out, W_out, stride,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_W=64,
    )

    return y


def linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Linear layer: y = x @ weight.T + bias
    x: [B, in_features]
    weight: [out_features, in_features]
    bias: [out_features]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, K = x.shape
    out_features, K_w = weight.shape
    assert K == K_w
    assert bias.shape[0] == out_features

    B_mat = weight.t().contiguous()  # [K, out_features]
    y = torch.empty((B, out_features), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_M']),
        triton.cdiv(out_features, META['BLOCK_N']),
    )

    fused_gemm_bias_kernel[grid](
        x, B_mat, bias, y,
        B, out_features, K,
        x.stride(0), x.stride(1),
        B_mat.stride(0), B_mat.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    return y


class MBConvBlockNew(nn.Module):
    """
    MBConv block using Triton-optimized 1x1 and depthwise 3x3 convolutions.
    Structure:
      1x1 conv (expand) -> BN -> ReLU6
      3x3 depthwise conv -> BN -> ReLU6
      1x1 conv (project) -> BN
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MBConvBlockNew, self).__init__()
        hidden_dim = round(in_channels * expand_ratio)

        # 1x1 expand
        self.expand_weight = nn.Parameter(
            torch.randn(hidden_dim, in_channels, 1, 1)
        )
        self.expand_bn = nn.BatchNorm2d(hidden_dim)

        # 3x3 depthwise
        self.depthwise_weight = nn.Parameter(
            torch.randn(hidden_dim, 1, 3, 3)
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.stride = stride

        # 1x1 project
        self.project_weight = nn.Parameter(
            torch.randn(out_channels, hidden_dim, 1, 1)
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1x1 expand
        x = conv1x1_triton(x, self.expand_weight)
        x = self.expand_bn(x)
        x = nn.functional.relu6(x, inplace=True)

        # 3x3 depthwise
        x = depthwise_conv3x3_triton(x, self.depthwise_weight, self.stride)
        x = self.depthwise_bn(x)
        x = nn.functional.relu6(x, inplace=True)

        # 1x1 project
        x = conv1x1_triton(x, self.project_weight)
        x = self.project_bn(x)

        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1-like architecture with Triton-optimized kernels
        for 1x1 convolutions, depthwise 3x3 convolutions, and the final
        fully connected layer.
        """
        super(ModelNew, self).__init__()

        # Initial convolutional layer (keep PyTorch's highly-optimized conv)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks (Triton-based internals)
        self.mbconv1 = MBConvBlockNew(32, 16, 1, 1)
        self.mbconv2 = MBConvBlockNew(16, 24, 2, 6)
        self.mbconv3 = MBConvBlockNew(24, 40, 2, 6)
        self.mbconv4 = MBConvBlockNew(40, 80, 2, 6)
        self.mbconv5 = MBConvBlockNew(80, 112, 1, 6)
        self.mbconv6 = MBConvBlockNew(112, 192, 2, 6)
        self.mbconv7 = MBConvBlockNew(192, 320, 1, 6)

        # Final 1x1 convolution (Triton)
        self.conv2_weight = nn.Parameter(
            torch.randn(1280, 320, 1, 1)
        )
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer (Triton)
        self.fc_weight = nn.Parameter(torch.randn(num_classes, 1280))
        self.fc_bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        # x: [batch_size, 3, 240, 240]
        x = nn.functional.relu(self.bn1(self.conv1(x)))

        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        # Final 1x1 conv via Triton
        x = conv1x1_triton(x, self.conv2_weight)
        x = nn.functional.relu(self.bn2(x))

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        x = linear_triton(x, self.fc_weight, self.fc_bias)
        return x
