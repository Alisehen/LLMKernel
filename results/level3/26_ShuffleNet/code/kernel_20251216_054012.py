import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),  # baseline
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    ],
    key=["numel"],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr,  # *f32 / *f16 / ...
    y_ptr,
    N, C, H, W,
    channels_per_group,
    numel,
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """
    Fast channel shuffle for NCHW tensors.

    Layout:
      - x, y: [N, C, H, W] contiguous (NCHW)

    We treat the flat index as:
      offs = ((n*C + c_out) * (H*W) + t)
    where t = h*W + w.

    Shuffle only changes the channel index (c_out -> c_in), so:
      q    = offs // (H*W)   = n*C + c_out
      t    = offs %  (H*W)
      c_out = q % C
      c_in  = group * channels_per_group + idx
      q_in  = q - c_out + c_in
      in_offs = q_in * (H*W) + t
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Spatial area per channel
    spatial = H * W  # H*W

    # Decompose flat index into (q, t) where:
    #   q = n*C + c_out
    #   t = h*W + w
    q = offs // spatial
    t = offs % spatial

    # Extract output channel index c_out
    c_out = q % C

    # Compute shuffle mapping on channels: c_out -> c_in
    group = c_out % GROUPS
    idx = c_out // GROUPS
    c_in = group * channels_per_group + idx

    # Replace c_out with c_in inside q
    q_in = q - c_out + c_in
    in_offs = q_in * spatial + t

    # Load / store
    x_vals = tl.load(x_ptr + in_offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x_vals, mask=mask)


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Fast Triton channel shuffle for NCHW tensors.

    Args:
        x: (N, C, H, W), contiguous CUDA tensor
        groups: number of channel groups (C % groups == 0)

    Returns:
        y: (N, C, H, W) with channels shuffled.
    """
    assert x.is_cuda, "Triton channel_shuffle requires CUDA tensor"
    x = x.contiguous()
    N, C, H, W = x.shape
    assert C % groups == 0, "C must be divisible by groups"
    channels_per_group = C // groups

    y = torch.empty_like(x)
    numel = x.numel()

    def grid(meta):
        return (triton.cdiv(numel, meta["BLOCK_SIZE"]),)

    channel_shuffle_kernel[grid](
        x, y,
        N, C, H, W,
        channels_per_group,
        numel,
        GROUPS=groups,
    )
    return y


class ChannelShuffleTriton(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to native implementation on CPU or non-contiguous tensors
        if (not x.is_cuda) or (not x.is_contiguous()):
            batch_size, channels, height, width = x.size()
            channels_per_group = channels // self.groups
            x = x.view(batch_size, self.groups, channels_per_group, height, width)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batch_size, -1, height, width)
            return x
        return channel_shuffle_triton(x, self.groups)


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit with Triton-accelerated channel shuffle.
        """
        super().__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

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

        # Triton-based shuffle
        self.shuffle = ChannelShuffleTriton(groups)

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

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = torch.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return out


class ModelNew(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        groups=3,
        stages_repeats=[3, 7, 3],
        stages_out_channels=[24, 240, 480, 960],
    ):
        """
        ShuffleNet architecture with Triton-accelerated channel shuffle.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            stages_out_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(
            stages_out_channels[0],
            stages_out_channels[1],
            stages_repeats[0],
            groups,
        )
        self.stage3 = self._make_stage(
            stages_out_channels[1],
            stages_out_channels[2],
            stages_repeats[1],
            groups,
        )
        self.stage4 = self._make_stage(
            stages_out_channels[2],
            stages_out_channels[3],
            stages_repeats[2],
            groups,
        )

        self.conv5 = nn.Conv2d(
            stages_out_channels[3],
            1024,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
