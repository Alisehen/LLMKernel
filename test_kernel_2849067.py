import torch
import torch.nn as nn
import triton
import triton.language as tl


# ===========================
# Triton Kernels
# ===========================

@triton.jit
def linear_matmul_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (weight^T)
    bias_ptr,  # [N] or dummy
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    acc = acc.to(tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def global_avg_pool2d_kernel(
    x_ptr,  # [N, C, H, W]
    y_ptr,  # [N, C]
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Each program handles one (n, c)
    n = pid // C
    c = pid % C

    base = n * stride_n + c * stride_c
    HW = H * W

    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over all H*W positions in tiles of BLOCK_SIZE
    for s in range(0, HW, BLOCK_SIZE):
        pos = s + offs
        mask = pos < HW
        h = pos // W
        w = pos % W
        ptrs = x_ptr + base + h * stride_h + w * stride_w
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    # HW is an integer kernel argument; Triton will promote it to float for division
    mean = total / HW

    # y is assumed contiguous [N, C] with stride (C, 1)
    out_offset = n * C + c
    tl.store(y_ptr + out_offset, mean)


# ===========================
# Triton Wrappers
# ===========================

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    """
    x: [M, K]
    weight: [N, K]  (PyTorch Linear: out_features x in_features)
    bias: [N] or None
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Triton linear requires CUDA tensors"
    M, K = x.shape
    N = weight.shape[0]
    b = weight.t().contiguous()  # [K, N]

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else c  # dummy pointer when no bias

    linear_matmul_bias_kernel[grid](
        x,
        b,
        bias_ptr,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return c


def triton_global_avg_pool2d(x: torch.Tensor):
    """
    x: [N, C, H, W] -> [N, C]
    """
    assert x.is_cuda, "Triton global avgpool requires CUDA tensor"
    N, C, H, W = x.shape
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # One program per (n, c)
    grid = (N * C,)

    global_avg_pool2d_kernel[grid](
        x,
        y,
        N,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return y


# ===========================
# EfficientNetB2 with Triton-optimized heads
# ===========================

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2-like architecture with Triton-optimized
        global average pooling and final linear layer.
        """
        super(ModelNew, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        # Final 1x1 conv + BN
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)

        # Triton-optimized head: global avg pool + linear
        self.fc_weight = nn.Parameter(torch.randn(num_classes, 1408))
        self.fc_bias = nn.Parameter(torch.randn(num_classes))

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Creates an MBConv block as in the original model.
        (Implemented with standard PyTorch layers for correctness;
         Triton is used for the global head where it matters most.)
        """
        layers = []
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        # Depthwise convolution
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels // 4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                expanded_channels // 4,
                expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.Sigmoid())

        # Output phase
        layers.append(
            nn.Conv2d(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the EfficientNetB2-like model with Triton-optimized head.

        x: (batch_size, 3, 224, 224)
        returns: (batch_size, num_classes)
        """
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))

        # MBConv stages
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)

        # Final conv + BN + ReLU
        x = self.relu(self.bn_final(self.conv_final(x)))  # [N, 1408, H, W]

        # Triton global avg pool -> [N, 1408]
        x = triton_global_avg_pool2d(x)

        # Triton linear -> [N, num_classes]
        x = triton_linear(x, self.fc_weight, self.fc_bias)
        return x
