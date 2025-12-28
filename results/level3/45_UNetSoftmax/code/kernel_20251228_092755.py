import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton Softmax over last dim
# -----------------------------

@triton.jit
def softmax_lastdim_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    # each program handles exactly one row, so grid must be M
    offs_n = tl.arange(0, BLOCK_N)

    # base pointers for this row
    row_x_ptr = x_ptr + pid_m * stride_xm
    row_y_ptr = y_ptr + pid_m * stride_ym

    # 1) compute row-wise max for numerical stability
    max_val = -float("inf")
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = idx < N
        x = tl.load(row_x_ptr + idx, mask=mask, other=-float("inf"))
        curr_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # 2) compute exp(x - max) and accumulate sum; store unnormalized to y
    sum_exp = 0.0
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = idx < N
        x = tl.load(row_x_ptr + idx, mask=mask, other=-float("inf"))
        x = x - max_val
        exp_x = tl.exp(x)
        tl.store(row_y_ptr + idx, exp_x, mask=mask)
        sum_exp = sum_exp + tl.sum(exp_x, axis=0)

    # 3) normalize
    inv_sum = 1.0 / sum_exp
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = idx < N
        y = tl.load(row_y_ptr + idx, mask=mask, other=0.0)
        y = y * inv_sum
        tl.store(row_y_ptr + idx, y, mask=mask)


def triton_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over last dimension (dim=-1), equivalent to nn.Softmax(dim=-1),
    implemented in Triton for high performance.

    Assumes x is CUDA and float32 (for numerical stability and simplicity).
    """
    assert x.is_cuda, "triton_softmax_lastdim requires a CUDA tensor"
    assert x.dtype == torch.float32, "triton_softmax_lastdim currently supports float32 only"

    *prefix, N = x.shape
    M = 1
    for s in prefix:
        M *= s

    x_2d = x.contiguous().view(M, N)
    y_2d = torch.empty_like(x_2d)

    stride_xm = x_2d.stride(0)
    stride_ym = y_2d.stride(0)

    def grid(meta):
        return (triton.cdiv(M, 1),)

    softmax_lastdim_kernel[grid](
        x_2d, y_2d,
        M, N,
        stride_xm, stride_ym,
        BLOCK_N=256,  # power-of-2 block size
    )
    return y_2d.view_as(x)


# -----------------------------
# Triton MaxPool2d (k=2, s=2)
# -----------------------------

@triton.jit
def maxpool2x2_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    H_out, W_out,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_out

    # Decode flattened output index -> (n, c, ho, wo)
    # idx = ((n*C + c)*H_out + ho)*W_out + wo
    idx = offs

    wo = idx % W_out
    tmp = idx // W_out
    ho = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    # Compute top-left corner (hi, wi) in input for each output element
    hi = ho * 2
    wi = wo * 2

    # Flattened input index for (n, c, hi, wi)
    # base = ((n*C + c)*H + hi)*W + wi
    base = ((n * C + c) * H + hi) * W + wi

    # four positions in 2x2 window
    idx00 = base
    idx01 = base + 1
    idx10 = base + W
    idx11 = base + W + 1

    # load with mask to avoid OOB if any (mainly for completeness)
    x00 = tl.load(x_ptr + idx00, mask=mask, other=-float("inf"))
    x01 = tl.load(x_ptr + idx01, mask=mask & (wi + 1 < W), other=-float("inf"))
    x10 = tl.load(x_ptr + idx10, mask=mask & (hi + 1 < H), other=-float("inf"))
    x11 = tl.load(
        x_ptr + idx11,
        mask=mask & (wi + 1 < W) & (hi + 1 < H),
        other=-float("inf"),
    )

    m1 = tl.maximum(x00, x01)
    m2 = tl.maximum(x10, x11)
    m = tl.maximum(m1, m2)

    tl.store(y_ptr + offs, m, mask=mask)


def triton_maxpool2x2(x: torch.Tensor) -> torch.Tensor:
    """
    MaxPool2d with kernel_size=2, stride=2, no padding.
    Equivalent to nn.MaxPool2d(2, 2).

    Assumes NCHW, CUDA tensor, contiguous.
    """
    assert x.is_cuda, "triton_maxpool2x2 requires a CUDA tensor"
    assert x.dim() == 4, "Expected NCHW input"
    x = x.contiguous()
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for k=2, s=2"

    H_out = H // 2
    W_out = W // 2
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    total_out = N * C * H_out * W_out

    def grid(meta):
        return (triton.cdiv(total_out, meta["BLOCK"]),)

    maxpool2x2_kernel[grid](
        x, y,
        N, C, H, W,
        H_out, W_out,
        total_out,
        BLOCK=256,
    )
    return y


# -----------------------------
# Network definitions
# -----------------------------

class DoubleConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Keep Conv2d + BatchNorm2d in PyTorch, replace Softmax with Triton
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = triton_softmax_lastdim(x)  # Softmax over last dim (width)
        x = self.conv2(x)
        x = self.bn2(x)
        x = triton_softmax_lastdim(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        U-Net-like architecture with Triton-optimized Softmax and MaxPool.
        """
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvNew(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvNew(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvNew(features * 8, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with Triton MaxPool2d
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(triton_maxpool2x2(enc1))
        enc3 = self.encoder3(triton_maxpool2x2(enc2))
        enc4 = self.encoder4(triton_maxpool2x2(enc3))

        bottleneck = self.bottleneck(triton_maxpool2x2(enc4))

        # Decoder path (uses PyTorch ConvTranspose2d + Triton-optimized blocks)
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
