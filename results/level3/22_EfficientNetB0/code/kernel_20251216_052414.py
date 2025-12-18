# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.nn.functional as F
import math


# =========================
# 1. Triton kernel(s)
# =========================

@triton.autotune(
    configs=[
        # Conservative baseline (multi-input fusion: 3 loads => stages=2)
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=4,
            num_stages=2,
        ),
        # More aggressive tiling in N, still 4 warps
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        # High-parallelism config, good when register pressure is low
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    bias_ptr,  # (N,) or dummy if HAS_BIAS = False
    c_ptr,  # (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,  # kept for autotune compatibility, not used in mapping
):
    """
    High-performance GEMM + bias kernel.

    A: [M, K] with strides (stride_am, stride_ak)
    B: [K, N] with strides (stride_bk, stride_bn)
    C: [M, N] with strides (stride_cm, stride_cn)
    """
    pid = tl.program_id(0)

    # Compute tile indices along M and N.
    # NOTE: We use a simple, correct 2D mapping from a 1D pid:
    #   pid_m = pid // num_pid_n
    #   pid_n = pid %  num_pid_n
    # This guarantees every (pid_m, pid_n) tile is covered exactly once,
    # avoiding the previous GROUP_M swizzling bug that skipped tiles.
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers for A and B; they will be advanced along K
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute masks that don't depend on K-iteration
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Reduction along K
    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        k_mask = offs_k[None, :] < k_rem

        a_mask = m_mask & k_mask
        b_mask = k_mask.T & n_mask

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Optional bias fused after GEMM
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Write back C, cast accumulator to output dtype on store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


# =========================
# 2. Wrapper function(s)
# =========================

def matmul_bias(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Fused matmul + bias using optimized Triton kernel.

    a:      (M, K)
    weight: (N, K)  [same layout as nn.Linear weight]
    bias:   (N,) or None
    returns: (M, N)
    """
    assert a.ndim == 2
    assert weight.ndim == 2
    M, K = a.shape
    N, K_w = weight.shape
    assert K == K_w, "Incompatible shapes for matmul_bias"

    # Triton kernel expects B as (K, N)
    a = a.contiguous()                  # (M, K)
    b = weight.t().contiguous()         # (K, N)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    HAS_BIAS = bias is not None
    if HAS_BIAS:
        bias_vec = bias.contiguous()
    else:
        # dummy tensor (never read when HAS_BIAS = False)
        bias_vec = c

    def grid(meta):
        BM, BN = meta["BLOCK_M"], meta["BLOCK_N"]
        num_pid_m = triton.cdiv(M, BM)
        num_pid_n = triton.cdiv(N, BN)
        # 1D launch; kernel maps pid -> (pid_m, pid_n) without swizzling
        return (num_pid_m * num_pid_n,)

    matmul_bias_kernel[grid](
        a, b, bias_vec, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=HAS_BIAS,
    )
    return c


def pointwise_conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Efficient 1x1 conv using Triton matmul:

    x:      (N, Cin, H, W)
    weight: (Cout, Cin, 1, 1)
    bias:   (Cout,) or None
    returns: (N, Cout, H, W)
    """
    assert x.ndim == 4
    assert weight.ndim == 4
    N, Cin, H, W = x.shape
    Cout, Cin_w, kh, kw = weight.shape
    assert kh == 1 and kw == 1, "pointwise_conv2d_triton only supports 1x1 conv"
    assert Cin == Cin_w

    # Flatten spatial dims; convert to (M, K) where M = N*H*W, K = Cin
    x_2d = x.permute(0, 2, 3, 1).reshape(-1, Cin).contiguous()  # (M, Cin)

    # Weight as (Cout, Cin)
    w_2d = weight.view(Cout, Cin).contiguous()  # (Cout, Cin)

    out_2d = matmul_bias(x_2d, w_2d, bias)  # (M, Cout)

    # Reshape back to NCHW
    out = out_2d.view(N, H, W, Cout).permute(0, 3, 1, 2).contiguous()
    return out


class LinearTriton(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Supports arbitrary batch shape ending in in_features
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        out_2d = matmul_bias(x_2d, self.weight, self.bias)
        out = out_2d.view(*orig_shape, self.out_features)
        return out


class PointwiseConv2dTriton(nn.Module):
    """
    1x1 convolution implemented via Triton matmul.
    Drop-in replacement for nn.Conv2d with kernel_size=1, stride=1, padding=0, groups=1.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in = self.in_channels
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pointwise_conv2d_triton(x, self.weight, self.bias)


class MBConvNew(nn.Module):
    """
    MBConv block implementation with Triton-accelerated 1x1 convolutions.
    Depthwise convolutions remain as PyTorch nn.Conv2d for simplicity and correctness.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        # Expansion 1x1 conv (if expand_ratio != 1), using Triton
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                PointwiseConv2dTriton(in_channels, hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )
        else:
            self.expand_conv = None

        # Depthwise conv (standard PyTorch conv, groups=hidden_dim)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        # Projection 1x1 conv, using Triton
        self.project_conv = nn.Sequential(
            PointwiseConv2dTriton(hidden_dim, out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.expand_conv is not None:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        if self.use_residual:
            x = x + identity

        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0-like architecture using Triton-optimized 1x1 convolutions and final linear layer.
        """
        super(ModelNew, self).__init__()

        # Initial convolutional layer (remain as standard Conv2d)
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks, using MBConvNew with Triton pointwise convs
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        # Final convolutional layer: 1x1 conv using Triton
        self.conv2 = PointwiseConv2dTriton(320, 1280, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer using Triton matmul
        self.fc = LinearTriton(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EfficientNetB0-style model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
