import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=1),
    ],
    key=["HW"],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr,  # *const T
    y_ptr,  # *mut T
    B,
    C,
    HW,
    groups,
    channels_per_group,
    BLOCK_HW: tl.constexpr,
):
    """
    Highly optimized channel shuffle kernel.

    Grid:
      axis 0: B * C  (each program = one (b, c) channel map)
      axis 1: ceil_div(HW, BLOCK_HW)  (each program processes a tile in HW)

    For each (b, c), we compute new_c once and copy the spatial tile [hw] contiguously:
        y[b, new_c, hw] = x[b, c, hw]
    """

    pid_bc = tl.program_id(axis=0)  # index over (b, c)
    pid_tile = tl.program_id(axis=1)  # index over tiles of HW

    # Derive batch and channel index from pid_bc
    b = pid_bc // C
    c = pid_bc % C

    # Compute shuffle mapping:
    #   original: c = g * channels_per_group + k
    #   shuffled: c' = k * groups + g
    g = c // channels_per_group
    k = c % channels_per_group
    new_c = k * groups + g

    # Base pointers (flattened NCHW -> [B*C, HW])
    base_x = pid_bc * HW
    base_y = (b * C + new_c) * HW

    # Offsets within spatial HW dimension for this tile
    offs_hw = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    # Coalesced load/store along HW
    x_vals = tl.load(x_ptr + base_x + offs_hw, mask=mask, other=0.0)
    tl.store(y_ptr + base_y + offs_hw, x_vals, mask=mask)


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Triton implementation of channel shuffle.

    x: (B, C, H, W) contiguous tensor on CUDA
    groups: number of channel groups
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"
    B, C, H, W = x.shape
    assert C % groups == 0, "Channels must be divisible by groups"

    HW = H * W
    y = torch.empty_like(x)
    channels_per_group = C // groups

    def grid(meta):
        return (B * C, triton.cdiv(HW, meta["BLOCK_HW"]))

    channel_shuffle_kernel[grid](
        x,
        y,
        B,
        C,
        HW,
        groups,
        channels_per_group,
    )
    return y


class ModelNew(nn.Module):
    """
    ShuffleNet unit with Triton-accelerated ChannelShuffle.
    Convolutions and batch norms remain as standard PyTorch modules
    for robustness and correctness.
    """

    def __init__(self, in_channels, out_channels, groups=3):
        """
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ModelNew, self).__init__()

        # Ensure the output channels are divisible by groups as in the original
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.groups = groups

        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Triton-accelerated channel shuffle (fully coalesced HW access)
        out = channel_shuffle_triton(out, self.groups)

        out = self.conv3(out)
        out = self.bn3(out)
        out = torch.relu(out)

        out = out + self.shortcut(x)
        return out
