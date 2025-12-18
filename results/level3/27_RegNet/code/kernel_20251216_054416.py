# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Larger tiles for high arithmetic intensity on large feature maps
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=2),
        # Extreme cases for very large / very small HW
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 32},  num_warps=1, num_stages=2),
    ],
    key=['HW_OUT'],
)
@triton.jit
def relu_maxpool2x2_kernel(
    x_ptr, y_ptr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C, HW_OUT, W_OUT,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused ReLU + 2x2 MaxPool (stride=2), NCHW layout, no padding.
    Semantics: y = max_pool2d(relu(x), kernel_size=2, stride=2)
    Implemented as: pooled = max_pool2d(x); y = relu(pooled)
    (mathematically equivalent for ReLU).
    """
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Decode (n, c) from flat index
    n = pid_nc // C
    c = pid_nc - n * C

    # Vector of output spatial indices this program handles
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW_OUT

    # H_out, W_out indices
    h_out = offs_hw // W_OUT
    w_out = offs_hw - h_out * W_OUT

    # Input top-left corner of 2x2 window
    h_in = h_out * 2
    w_in = w_out * 2

    # Help the compiler with alignment / contiguity assumptions on the fast axis
    tl.multiple_of(offs_hw, BLOCK_HW)
    tl.max_contiguous(offs_hw, BLOCK_HW)

    # Base pointer for each 2x2 window (top-left element)
    x_base = (
        x_ptr
        + n * stride_xn
        + c * stride_xc
        + h_in * stride_xh
        + w_in * stride_xw
    )

    # 2x2 window pointers (row-major)
    row0 = x_base
    row1 = x_base + stride_xh

    # Streaming loads: each input element is used exactly once for K=2,S=2,
    # so bypass L1 (cg) and rely on L2/global bandwidth.
    x00 = tl.load(row0,                      mask=mask, other=0.0, cache_modifier='.cg')
    x01 = tl.load(row0 + stride_xw,          mask=mask, other=0.0, cache_modifier='.cg')
    x10 = tl.load(row1,                      mask=mask, other=0.0, cache_modifier='.cg')
    x11 = tl.load(row1 + stride_xw,          mask=mask, other=0.0, cache_modifier='.cg')

    # 2x2 max-pool in registers
    m0 = tl.maximum(x00, x01)
    m1 = tl.maximum(x10, x11)
    pooled = tl.maximum(m0, m1)

    # Fused ReLU (still in registers, no intermediate stores)
    y_val = tl.maximum(pooled, 0.0)

    # Output pointer y[n, c, h_out, w_out]
    y_ptrs = (
        y_ptr
        + n * stride_yn
        + c * stride_yc
        + h_out * stride_yh
        + w_out * stride_yw
    )

    # Single final store -> fusion-friendly
    tl.store(y_ptrs, y_val, mask=mask)


def fused_relu_maxpool2x2(x: torch.Tensor) -> torch.Tensor:
    """
    Fused ReLU + 2x2 MaxPool (stride=2) for NCHW tensors using Triton.
    Replaces:
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
    """
    if not x.is_cuda:
        return torch.nn.functional.max_pool2d(
            torch.nn.functional.relu(x),
            kernel_size=2,
            stride=2,
        )

    # Guarantee dense NCHW for optimal memory pattern
    x = x.contiguous()
    N, C, H, W = x.shape
    H_out = H // 2
    W_out = W // 2
    HW_OUT = H_out * W_out

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    NC = N * C

    def grid(meta):
        # One program per (n, c) pair, tiling over flattened H_out * W_out
        return (
            NC,
            triton.cdiv(HW_OUT, meta['BLOCK_HW']),
        )

    relu_maxpool2x2_kernel[grid](
        x, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        C, HW_OUT, W_out,
    )
    return y


class StageNew(nn.Module):
    """
    One stage of the network:
        Conv2d -> BatchNorm2d -> ReLU ->
        Conv2d -> BatchNorm2d -> (fused ReLU + MaxPool2d)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(StageNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = fused_relu_maxpool2x2(x)  # Triton-accelerated fused kernel
        return x


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet-like architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(StageNew(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        x = self.feature_extractor(x)
        # Global Average Pooling over spatial dimensions
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x
