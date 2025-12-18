import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, y_ptr,
    N, C, HW,
    groups, channels_per_group,
    BLOCK_HW: tl.constexpr,
):
    """
    Optimized channel shuffle kernel.

    Layout view:
      - Treat input as 2D [K, HW] where K = N*C, HW = H*W.
      - For each row k_out = n*C + c_out and column u in [0, HW):
          y[k_out, u] = x[k_in, u]
        with channel mapping c_out -> c_in (shuffle within C).
    """

    pid_k = tl.program_id(0)  # index over [0, N*C)
    pid_u = tl.program_id(1)  # tiles along HW dimension

    # Column indices for this HW-tile
    u = pid_u * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw = HW

    # Bounds masks
    k_bound = N * C
    mask = (u < hw) & (pid_k < k_bound)

    # Scalar row index (broadcast across the vector)
    k_out = pid_k

    # Decode k_out -> (n, c_out)
    C_i32 = C
    n = k_out // C_i32
    c_out = k_out - n * C_i32

    # Channel shuffle: c_out -> c_in
    # c_out = idx * groups + group
    group = c_out % groups
    idx = c_out // groups
    c_in = group * channels_per_group + idx

    # Re-encode (n, c_in) -> k_in
    k_in = n * C_i32 + c_in

    # Flat offsets in NCHW layout via [K, HW] view:
    # off = u + HW * k
    out_offsets = k_out * hw + u
    in_offsets = k_in * hw + u

    x_vals = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(y_ptr + out_offsets, x_vals, mask=mask)


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    High-performance channel shuffle using Triton.

    Input/Output: (N, C, H, W), with C % groups == 0.
    Falls back to native PyTorch implementation for non-CUDA or non-contiguous tensors.
    """
    if (not x.is_cuda) or (not x.is_contiguous()):
        n, c, h, w = x.size()
        assert c % groups == 0
        channels_per_group = c // groups
        x = x.view(n, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(n, -1, h, w)
        return x

    assert x.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    assert C % groups == 0
    channels_per_group = C // groups
    HW = H * W

    # For typical ShuffleNet shapes (e.g., ImageNet), N*C*H*W fits in int32.
    # If extremely large tensors are used, consider switching to int64 indexing.
    y = torch.empty_like(x)

    def grid(meta):
        # 2D grid:
        #  - dim0 over K = N*C (rows)
        #  - dim1 over HW (columns) in tiles of BLOCK_HW
        return (
            N * C,
            triton.cdiv(HW, meta['BLOCK_HW']),
        )

    channel_shuffle_kernel[grid](
        x, y,
        N, C, HW,
        groups, channels_per_group,
    )
    return y


class ModelNew(nn.Module):
    """
    ShuffleNet-style model with an aggressively optimized Triton-based channel shuffle.
    """

    class ShuffleNetUnit(nn.Module):
        def __init__(self, in_channels, out_channels, groups: int):
            super().__init__()
            assert out_channels % 4 == 0
            mid_channels = out_channels // 4
            self.groups = groups

            # First 1x1 group convolution
            self.conv1 = nn.Conv2d(
                in_channels, mid_channels,
                kernel_size=1, stride=1, padding=0,
                groups=groups, bias=False,
            )
            self.bn1 = nn.BatchNorm2d(mid_channels)

            # Depthwise 3x3 convolution
            self.conv2 = nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                groups=mid_channels, bias=False,
            )
            self.bn2 = nn.BatchNorm2d(mid_channels)

            # Second 1x1 group convolution
            self.conv3 = nn.Conv2d(
                mid_channels, out_channels,
                kernel_size=1, stride=1, padding=0,
                groups=groups, bias=False,
            )
            self.bn3 = nn.BatchNorm2d(out_channels)

            # Shortcut connection
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels,
                        kernel_size=1, stride=1, padding=0, bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = channel_shuffle_triton(out, self.groups)
            out = torch.nn.functional.relu(self.bn3(self.conv3(out)))
            out = out + self.shortcut(x)
            return out

    def __init__(
        self,
        num_classes: int = 1000,
        groups: int = 3,
        stages_repeats=(3, 7, 3),
        stages_out_channels=(24, 240, 480, 960),
    ):
        super().__init__()

        self.groups = groups

        # Initial conv + maxpool
        self.conv1 = nn.Conv2d(
            3, stages_out_channels[0],
            kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ShuffleNet stages
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

        # Final 1x1 conv
        self.conv5 = nn.Conv2d(
            stages_out_channels[3],
            1024,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.bn5 = nn.BatchNorm2d(1024)

        # Classifier
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ModelNew.ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ModelNew.ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = torch.nn.functional.relu(self.bn5(self.conv5(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
