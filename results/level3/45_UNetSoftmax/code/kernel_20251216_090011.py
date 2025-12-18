# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline (good for most sizes)
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        # Smaller tile for very small last-dim
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        # Aggressive config for large N when register pressure is low
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def softmax_lastdim_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    """
    Softmax along the last dimension (size N) for a 2D tensor of shape (M, N).

    - 1 program instance per row (M).
    - For N <= BLOCK_N: single-pass softmax (max + sum + normalize).
    - For N  > BLOCK_N: 2-pass streaming log-sum-exp algorithm.
    """

    m = tl.program_id(0)
    if m >= M:
        return

    # Pointers to the start of the m-th row
    x_row_ptr = x_ptr + m * stride_xm
    y_row_ptr = y_ptr + m * stride_ym

    # Precompute lane offsets once; reused across all paths
    lane_offsets = tl.arange(0, BLOCK_N)

    # -------------------------
    # Fast path: whole row fits in a single block
    # -------------------------
    if N <= BLOCK_N:
        offs = lane_offsets
        mask = offs < N

        # x is always float32 (wrapper guarantees it)
        x = tl.load(
            x_row_ptr + offs * stride_xn,
            mask=mask,
            other=-float('inf'),
        )

        row_max = tl.max(x, axis=0)
        x = x - row_max
        num = tl.exp(x)
        num = tl.where(mask, num, 0.0)
        row_sum = tl.sum(num, axis=0)
        out = num / row_sum

        tl.store(
            y_row_ptr + offs * stride_yn,
            out,
            mask=mask,
        )
        return

    # -------------------------
    # Streaming path: 2-pass LSE for large N
    # -------------------------
    row_max = tl.full((), -float('inf'), tl.float32)
    row_sum = tl.zeros((), tl.float32)

    # Pass 1: compute row_max and row_sum via streaming log-sum-exp
    n_start = 0
    while n_start < N:
        offs = n_start + lane_offsets
        mask = offs < N

        x = tl.load(
            x_row_ptr + offs * stride_xn,
            mask=mask,
            other=-float('inf'),
        )

        block_max = tl.max(x, axis=0)
        new_row_max = tl.maximum(row_max, block_max)

        # Rescale old sum to new max frame
        scale_old = tl.exp(row_max - new_row_max)
        # New block contribution in new max frame
        block_exp = tl.exp(x - new_row_max)
        block_sum = tl.sum(block_exp, axis=0)

        row_sum = row_sum * scale_old + block_sum
        row_max = new_row_max

        n_start += BLOCK_N

    # Pass 2: compute normalized softmax outputs
    inv_row_sum = 1.0 / row_sum
    n_start = 0
    while n_start < N:
        offs = n_start + lane_offsets
        mask = offs < N

        x = tl.load(
            x_row_ptr + offs * stride_xn,
            mask=mask,
            other=-float('inf'),
        )

        exp_x = tl.exp(x - row_max)
        softmax_x = exp_x * inv_row_sum

        tl.store(
            y_row_ptr + offs * stride_yn,
            softmax_x,
            mask=mask,
        )

        n_start += BLOCK_N


def triton_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over the last dimension of x using an optimized Triton kernel.

    x: (..., W)
    Returns tensor with same shape and dtype as x.
    """
    if x.numel() == 0:
        return x

    orig_dtype = x.dtype

    # Compute in float32 for numerical stability; keep layout contiguous
    x_f = x.to(torch.float32).contiguous()
    last_dim = x_f.shape[-1]
    x_2d = x_f.view(-1, last_dim)
    M, N = x_2d.shape

    y_2d = torch.empty_like(x_2d)

    grid = lambda META: (M,)

    softmax_lastdim_kernel[grid](
        x_2d, y_2d,
        M, N,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
    )

    y = y_2d.view_as(x_f)
    if orig_dtype != torch.float32:
        y = y.to(orig_dtype)
    return y


class ModelNew(nn.Module):
    """
    U-Net-like model with Triton-accelerated Softmax in DoubleConv blocks.
    Softmax is applied over the last dimension (width) of the feature maps.
    """

    class DoubleConv(nn.Module):
        """
        Two consecutive Conv2d + BatchNorm2d + Softmax(dim=-1),
        with Softmax implemented via an optimized Triton kernel.
        """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            # First conv block
            x = self.conv1(x)
            x = self.bn1(x)
            x = triton_softmax_lastdim(x)  # softmax over width (last dim)

            # Second conv block
            x = self.conv2(x)
            x = self.bn2(x)
            x = triton_softmax_lastdim(x)  # softmax over width (last dim)
            return x

    def __init__(self, in_channels, out_channels, features):
        """
        U-Net-like model with Triton-accelerated Softmax in DoubleConv blocks.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (doubled in each encoder layer)
        """
        super(ModelNew, self).__init__()

        DoubleConv = ModelNew.DoubleConv

        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
