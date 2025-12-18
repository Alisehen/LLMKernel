import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tiles for small HW / tiny feature maps
        triton.Config({"BLOCK_HW": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        # Good general-purpose tiles
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        # Aggressive large tile for very large HW
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr,  # *const T
    y_ptr,  # *mut T
    B,      # int32
    C,      # int32
    HW,     # int32
    groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fast channel shuffle kernel.

    Layout: x, y are contiguous (B, C, H, W) tensors flattened to [B*C, HW].

    Grid:
      axis 0: B * C      (each program handles a single (b, c) channel)
      axis 1: ceil_div(HW, BLOCK_HW)  (tiles along contiguous HW dimension)

    Each program:
      y[b, new_c, hw_tile] = x[b, c, hw_tile]

    Single global store:
      - Only one tl.store() for the final output.
      - All index math and mapping stay in registers.
    """
    pid_bc = tl.program_id(axis=0)   # 0 .. B*C-1, maps to (b, c)
    pid_tile = tl.program_id(axis=1) # 0 .. ceil_div(HW, BLOCK_HW)-1

    # Derive batch and channel indices
    C_i32 = tl.cast(C, tl.int32)
    b = pid_bc // C_i32
    c = pid_bc % C_i32

    # Channel shuffle mapping (all scalar, kept in registers):
    #   c = g * channels_per_group + k
    #   new_c = k * groups + g
    g = c // channels_per_group
    k = c % channels_per_group
    new_c = k * groups + g

    # Base indices in flattened [B*C, HW] layout
    HW_i32 = tl.cast(HW, tl.int32)
    bc = b * C_i32 + c
    bc_new = b * C_i32 + new_c
    base_x = bc * HW_i32
    base_y = bc_new * HW_i32

    # Offsets along contiguous HW dimension
    offs_hw = pid_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW_i32

    # Help compiler reason about alignment / vectorization
    tl.multiple_of(offs_hw, BLOCK_HW)

    # Fully coalesced load/store along HW (single final store)
    x_vals = tl.load(x_ptr + base_x + offs_hw, mask=mask, other=0.0)
    tl.store(y_ptr + base_y + offs_hw, x_vals, mask=mask)


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Triton-accelerated channel shuffle for (B, C, H, W) contiguous CUDA tensors.

    - Pure permutation: no intermediate global stores, only one final write.
    - Memory pattern: fully coalesced along HW for both load and store.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.is_contiguous(), "Input must be contiguous"
    B, C, H, W = x.shape
    assert C % groups == 0, "Channels must be divisible by groups"

    # If groups == 1, shuffling is a no-op; avoid kernel launch.
    if groups == 1:
        return x

    HW = H * W
    y = torch.empty_like(x)
    channels_per_group = C // groups

    def grid(meta):
        # pid_bc in [0, B*C), pid_tile in [0, ceil_div(HW, BLOCK_HW))
        return (B * C, triton.cdiv(HW, meta["BLOCK_HW"]))

    channel_shuffle_kernel[grid](
        x,
        y,
        B,
        C,
        HW,
        groups=groups,
        channels_per_group=channels_per_group,
    )
    return y


class ModelNew(nn.Module):
    """
    ShuffleNet unit with Triton-accelerated ChannelShuffle.
    Convolutions and batch norms are kept as PyTorch modules.
    """

    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()

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

        # Shortcut connection
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Triton-accelerated channel shuffle: single global write, fully coalesced
        out = channel_shuffle_triton(out, self.groups)

        out = self.conv3(out)
        out = self.bn3(out)
        out = torch.relu(out)

        out = out + self.shortcut(x)
        return out
