# complete ModelNew code with optimized Triton kernels

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=2, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def residual_add_relu_kernel(
    x_ptr,  # pointer to main path tensor
    y_ptr,  # pointer to shortcut tensor
    out_ptr,  # pointer to output tensor
    N,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    z = x + y
    z = tl.maximum(z, 0.0)

    tl.store(out_ptr + offsets, z, mask=mask)


def residual_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fused residual add + ReLU:
        out = ReLU(x + y)
    x, y: tensors of the same shape, on CUDA.
    """
    # Ensure contiguity for simple linear addressing
    x_c = x.contiguous()
    y_c = y.contiguous()
    assert x_c.shape == y_c.shape, "Input tensors must have the same shape"

    out = torch.empty_like(x_c)
    N = out.numel()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

    residual_add_relu_kernel[grid](
        x_c,
        y_c,
        out,
        N,
    )
    return out


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused residual add + ReLU in Triton
        out = residual_add_relu(out, identity)

        return out
