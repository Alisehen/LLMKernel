# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------
# Triton math helpers (tanh/sigmoid/gelu/silu/mish/softmax)
# ---------------------------------------------------------


def _tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


def _tl_tanh(x):
    # 2*sigmoid(2x) - 1
    return 2.0 * _tl_sigmoid(2.0 * x) - 1.0


def _tl_gelu(x):
    # Approximate GELU (Hendrycks & Gimpel)
    c = 0.7978845608028654  # sqrt(2/pi)
    k = 0.044715
    return 0.5 * x * (1.0 + _tl_tanh(c * (x + k * x * x * x)))


def _tl_silu(x):
    return x * _tl_sigmoid(x)


def _tl_softmax(x, axis=0):
    x_max = tl.max(x, axis=axis)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=axis)
    return x_exp / x_sum


def _tl_mish(x):
    softplus = tl.log(1.0 + tl.exp(x))
    return x * _tl_tanh(softplus)


tl.sigmoid = _tl_sigmoid
tl.tanh = _tl_tanh
tl.gelu = _tl_gelu
tl.silu = _tl_silu
tl.softmax = _tl_softmax
tl.mish = _tl_mish


# -----------------------------
# Triton Softmax over width (dim = -1) in-place on NCHW
# -----------------------------

@triton.jit
def softmax_width_inplace_kernel(
    x_ptr,
    M,  # number of rows = N * C * H
    N,  # width dimension (W)
    stride_xm,
    BLOCK_N: tl.constexpr,
):
    """
    In-place softmax along the last dimension (width) for a tensor
    logically viewed as [M, N], with contiguous rows of length N.

    x_ptr: pointer to float32 array with shape [M, N] (row-major)
    """
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    row_mask = pid_m < M
    base = pid_m * stride_xm

    # 1) Compute row-wise max for numerical stability
    max_val = -float("inf")
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = row_mask & (idx < N)
        x = tl.load(x_ptr + base + idx, mask=mask, other=-float("inf"))
        curr_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # 2) Compute sum(exp(x - max))
    sum_exp = 0.0
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = row_mask & (idx < N)
        x = tl.load(x_ptr + base + idx, mask=mask, other=-float("inf"))
        x = x - max_val
        exp_x = tl.exp(x)
        sum_exp = sum_exp + tl.sum(exp_x, axis=0)

    inv_sum = 1.0 / sum_exp

    # 3) Normalize and write back in-place
    for start_n in range(0, N, BLOCK_N):
        idx = start_n + offs_n
        mask = row_mask & (idx < N)
        x = tl.load(x_ptr + base + idx, mask=mask, other=-float("inf"))
        x = x - max_val
        exp_x = tl.exp(x) * inv_sum
        tl.store(x_ptr + base + idx, exp_x, mask=mask)


def triton_softmax_width_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    In-place softmax along width (dim = -1) for NCHW tensors.

    Equivalent to:
        x = F.softmax(x, dim=-1)
    but implemented with a fused Triton kernel and no extra allocations.

    Assumes:
        - x is CUDA
        - x is float32
        - x is contiguous NCHW
    """
    assert x.is_cuda, "triton_softmax_width_inplace requires a CUDA tensor"
    assert x.dtype == torch.float32, "triton_softmax_width_inplace supports float32 only"
    assert x.dim() == 4, "Expected NCHW 4D tensor"

    if x.numel() == 0:
        return x

    x = x.contiguous()
    N, C, H, W = x.shape

    # Flatten logical view to [M, W] without creating a new tensor
    M = N * C * H
    stride_xm = W  # row stride in elements for contiguous NCHW

    def grid(meta):
        # One program per row
        return (max(1, M),)

    softmax_width_inplace_kernel[grid](
        x,
        M,
        W,
        stride_xm,
        BLOCK_N=256,  # power-of-2
    )
    return x


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

    idx = offs

    wo = idx % W_out
    tmp = idx // W_out
    ho = tmp % H_out
    tmp = tmp // H_out
    c = tmp % C
    n = tmp // C

    hi = ho * 2
    wi = wo * 2

    base = ((n * C + c) * H + hi) * W + wi

    idx00 = base
    idx01 = base + 1
    idx10 = base + W
    idx11 = base + W + 1

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
    Equivalent to nn.MaxPool2d(2, 2) for NCHW tensors.
    """
    assert x.is_cuda, "triton_maxpool2x2 requires a CUDA tensor"
    assert x.dim() == 4, "Expected NCHW input"
    x = x.contiguous()
    N, C, H, W = x.shape

    if N == 0 or C == 0 or H == 0 or W == 0:
        return x.new_empty((N, C, H // 2, W // 2))

    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for k=2, s=2"

    H_out = H // 2
    W_out = W // 2
    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    total_out = N * C * H_out * W_out

    def grid(meta):
        return (max(1, triton.cdiv(total_out, meta["BLOCK"])),)

    maxpool2x2_kernel[grid](
        x, y,
        N, C, H, W,
        H_out, W_out,
        total_out,
        BLOCK=256,
    )
    return y


# -----------------------------
# Conv+BatchNorm2d folding (eval-time fusion)
# -----------------------------


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d + BatchNorm2d into a single Conv2d for inference.

    The resulting conv produces the same output as conv -> bn
    when bn is in eval mode (using running_mean/var).
    """
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(bn, nn.BatchNorm2d)

    # Create a new conv with bias
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device, conv.weight.dtype)

    # Prepare parameters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    if conv.bias is not None:
        b_conv = conv.bias.clone()
    else:
        b_conv = torch.zeros(conv.out_channels, device=w_conv.device, dtype=w_conv.dtype)

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    inv_std = 1.0 / torch.sqrt(var + eps)

    # w_fused[c, :] = w_conv[c, :] * (gamma[c] * inv_std[c])
    scale = (gamma * inv_std).reshape(-1, 1)
    w_fused = w_conv * scale
    # b_fused[c] = beta[c] + gamma[c] * (b_conv[c] - mean[c]) * inv_std[c]
    b_fused = beta + (b_conv - mean) * gamma * inv_std

    fused_conv.weight.data.copy_(w_fused.view_as(conv.weight))
    fused_conv.bias.data.copy_(b_fused)
    fused_conv.requires_grad_(False)
    return fused_conv


# -----------------------------
# Network definitions
# -----------------------------


class DoubleConvNew(nn.Module):
    """
    DoubleConv block with:
        Conv2d -> BatchNorm2d -> Softmax(dim=-1) ->
        Conv2d -> BatchNorm2d -> Softmax(dim=-1)

    Optimizations:
      * In eval mode, BatchNorm2d layers are folded into the preceding Conv2d
        (Conv+BN fusion), eliminating BN from the forward path.
      * Softmax over width is implemented by a Triton in-place kernel
        without extra allocations or reshaping.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Original Conv + BN modules (used for training / for parameter storage)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Fused versions used only in eval mode
        self._fused = False
        self.fused_conv1: nn.Conv2d | None = None
        self.fused_conv2: nn.Conv2d | None = None

    def _maybe_fuse_eval(self):
        # Lazily fuse Conv+BN the first time we run in eval mode
        if self._fused:
            return
        # Ensure BN uses running stats at fusion time
        self.bn1.eval()
        self.bn2.eval()
        self.fused_conv1 = fuse_conv_bn_eval(self.conv1, self.bn1)
        self.fused_conv2 = fuse_conv_bn_eval(self.conv2, self.bn2)
        self._fused = True

    def forward(self, x):
        if self.training:
            # Training path: keep full Conv+BN for correctness
            x = self.conv1(x)
            x = self.bn1(x)
            x = triton_softmax_width_inplace(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = triton_softmax_width_inplace(x)
        else:
            # Inference path: use fused Conv (Conv+BN folded) + Triton softmax
            self._maybe_fuse_eval()
            x = self.fused_conv1(x)
            x = triton_softmax_width_inplace(x)
            x = self.fused_conv2(x)
            x = triton_softmax_width_inplace(x)
        return x


class ModelNew(nn.Module):
    """
    U-Net-like architecture with:
      * Conv+BN folded in DoubleConv blocks during eval
      * Triton-optimized softmax over width
      * Triton-optimized MaxPool2d with k=2, s=2
    """

    def __init__(self, in_channels, out_channels, features):
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = DoubleConvNew(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConvNew(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        # Correct in_channels after concatenation: features*2 (enc2) + features*2 (upconv2)
        self.decoder2 = DoubleConvNew(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with Triton MaxPool2d
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(triton_maxpool2x2(enc1))
        enc3 = self.encoder3(triton_maxpool2x2(enc2))
        enc4 = self.encoder4(triton_maxpool2x2(enc3))

        bottleneck = self.bottleneck(triton_maxpool2x2(enc4))

        # Decoder path
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
