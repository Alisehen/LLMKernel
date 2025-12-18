import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_relu_kernel(
    x_ptr,  # input: (B, Cin, H, W), contiguous NCHW
    w_ptr,  # weight: (Cin, Cout) as row-major (K, N)
    bias_ptr,  # bias: (Cout,)
    y_ptr,  # output: (B, Cout, H, W), contiguous NCHW

    M,  # total number of output pixels = B * H * W
    N,  # Cout
    K,  # Cin

    H,  # height
    W,  # width
    HW,  # H * W

    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,

    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,

    stride_w_k,  # weight_t.stride(0)
    stride_w_n,  # weight_t.stride(1)

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M (rows) and N (cols)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for valid rows/cols
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Decode flattened row index m -> (b, h, w)
    # m in [0, B*H*W)
    b_idx = offs_m // HW
    rem = offs_m - b_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    # Base offsets for input/output for channel = 0
    base_in = (
        b_idx * stride_in_n
        + h_idx * stride_in_h
        + w_idx * stride_in_w
    )
    base_out = (
        b_idx * stride_out_n
        + h_idx * stride_out_h
        + w_idx * stride_out_w
    )

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < K

        # A tile: from input x, shape (BLOCK_M, BLOCK_K)
        a_ptrs = x_ptr + base_in[:, None] + k_idx[None, :] * stride_in_c
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a = a.to(tl.float32)

        # B tile: from weight_t, shape (BLOCK_K, BLOCK_N)
        b_ptrs = w_ptr + k_idx[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        b = b.to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    bias = bias.to(tl.float32)
    acc += bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store results into y
    y_ptrs = y_ptr + base_out[:, None] + offs_n[None, :] * stride_out_c
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv1x1_bias_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused 1x1 convolution + bias + ReLU using Triton.

    Args:
        x: (B, Cin, H, W), contiguous NCHW.
        weight: (Cout, Cin, 1, 1) or (Cout, Cin).
        bias: (Cout,)

    Returns:
        y: (B, Cout, H, W)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "Only float32 is supported for this seed implementation"

    B, Cin, H, W = x.shape
    if weight.dim() == 4:
        Cout = weight.shape[0]
        assert weight.shape[1] == Cin and weight.shape[2] == 1 and weight.shape[3] == 1
        w_2d = weight.view(Cout, Cin)
    else:
        Cout, Cin_w = weight.shape
        assert Cin_w == Cin
        w_2d = weight

    M = B * H * W  # number of output pixels
    N = Cout
    K = Cin

    # Prepare output tensor
    y = torch.empty((B, Cout, H, W), device=x.device, dtype=x.dtype)

    # Transpose weight to (K, N) and make contiguous
    w_t = w_2d.t().contiguous()  # (Cin, Cout)

    # Strides
    stride_in_n, stride_in_c, stride_in_h, stride_in_w = x.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = y.stride()
    stride_w_k, stride_w_n = w_t.stride()

    HW = H * W

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    conv1x1_bias_relu_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        H, W, HW,
        stride_in_n, stride_in_c, stride_in_h, stride_in_w,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        stride_w_k, stride_w_n,
    )
    return y


class FireModuleTriton(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        FireModule with Triton-optimized 1x1 convolutions (squeeze and expand1x1).
        3x3 path uses standard PyTorch Conv2d + ReLU.
        """
        super(FireModuleTriton, self).__init__()

        # Squeeze: 1x1 conv
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        # Expand 1x1 path
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        # Expand 3x3 path
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Squeeze 1x1 + ReLU (fused in Triton)
        x = conv1x1_bias_relu_triton(x, self.squeeze.weight, self.squeeze.bias)

        # Expand 1x1 + ReLU (fused in Triton)
        out1 = conv1x1_bias_relu_triton(x, self.expand1x1.weight, self.expand1x1.bias)

        # Expand 3x3 + ReLU (PyTorch)
        out3 = F.relu(self.expand3x3(x), inplace=False)

        # Concatenate along channel dimension
        return torch.cat([out1, out3], dim=1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        SqueezeNet-like model with Triton-accelerated 1x1 convolutions
        in Fire modules and classifier.
        """
        super(ModelNew, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModuleTriton(96, 16, 64, 64),
            FireModuleTriton(128, 16, 64, 64),
            FireModuleTriton(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModuleTriton(256, 32, 128, 128),
            FireModuleTriton(256, 48, 192, 192),
            FireModuleTriton(384, 48, 192, 192),
            FireModuleTriton(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),

            FireModuleTriton(512, 64, 256, 256),
        )

        # Classifier parts broken out for manual Triton usage
        self.dropout = nn.Dropout(p=0.0)
        self.class_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.features(x)

        x = self.dropout(x)
        # Classifier 1x1 conv + bias + ReLU via Triton
        x = conv1x1_bias_relu_triton(x, self.class_conv.weight, self.class_conv.bias)
        x = self.avgpool(x)

        return torch.flatten(x, 1)
