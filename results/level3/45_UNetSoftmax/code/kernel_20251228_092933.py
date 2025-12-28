import torch, torch.nn as nn, triton, triton.language as tl

# ============================
# Triton kernels and wrappers
# ============================

@triton.jit
def softmax_lastdim_kernel(
    inp_ptr, out_ptr,
    ROWS, COLS,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax over the last dimension (length = COLS) for ROWS rows.
    Each program instance processes one full row and keeps it on-chip.

    This kernel is optimized for COLS <= 2 * BLOCK_SIZE (here: up to 512
    when BLOCK_SIZE=256), which matches the target workload (width=512).
    """

    row_id = tl.program_id(0)
    row_mask = row_id < ROWS

    # If row_id is out of range, we still compute with a dummy row_start=0 but
    # fully masked out, so no invalid memory accesses occur.
    safe_row = tl.where(row_mask, row_id, 0)
    row_start = safe_row * COLS

    offs0 = tl.arange(0, BLOCK_SIZE)
    offs1 = offs0 + BLOCK_SIZE

    mask0 = row_mask & (offs0 < COLS)
    mask1 = row_mask & (offs1 < COLS)

    # Load the entire row into two on-chip chunks (up to 2 * BLOCK_SIZE elements)
    x0 = tl.load(inp_ptr + row_start + offs0, mask=mask0, other=-1.0e30)
    x1 = tl.load(inp_ptr + row_start + offs1, mask=mask1, other=-1.0e30)

    # Row-wise max for numerical stability
    row_max0 = tl.max(x0, axis=0)
    row_max1 = tl.max(x1, axis=0)
    row_max = tl.maximum(row_max0, row_max1)

    # Compute exp(x - max) in-place in registers
    x0 = tl.exp(x0 - row_max)
    x1 = tl.exp(x1 - row_max)

    # Row-wise sum of numerators
    row_sum0 = tl.sum(x0, axis=0)
    row_sum1 = tl.sum(x1, axis=0)
    row_sum = row_sum0 + row_sum1

    inv_sum = 1.0 / row_sum

    # Normalize in-place and write final result (single global write)
    out0 = x0 * inv_sum
    out1 = x1 * inv_sum

    tl.store(out_ptr + row_start + offs0, out0, mask=mask0)
    tl.store(out_ptr + row_start + offs1, out1, mask=mask1)


def triton_softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax over last dimension (dim=-1) using an optimized Triton kernel.
    Works for N-D tensors; only last dimension is reduced.

    For this optimized implementation, the last dimension must satisfy
    last_dim <= 2 * BLOCK_SIZE (i.e., <= 512 when BLOCK_SIZE=256),
    which matches the target U-Net workload (width=512).
    """
    assert x.is_cuda, "Triton softmax requires CUDA tensor"
    x_contig = x.contiguous()
    orig_dtype = x_contig.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported dtype {orig_dtype} for Triton softmax")

    # Compute in fp32 for numerical stability
    if x_contig.dtype != torch.float32:
        x_ = x_contig.to(torch.float32)
    else:
        x_ = x_contig

    *leading, last = x_.shape
    rows = 1
    for d in leading:
        rows *= d
    cols = last

    # This kernel is designed for cols <= 512 (with BLOCK_SIZE=256)
    BLOCK_SIZE = 256
    if cols > 2 * BLOCK_SIZE:
        raise ValueError(
            f"Triton softmax kernel supports last dimension up to {2 * BLOCK_SIZE}, "
            f"got {cols}"
        )

    inp_flat = x_.view(rows, cols)
    out_flat = torch.empty_like(inp_flat)

    grid = lambda META: (triton.cdiv(rows, 1),)
    softmax_lastdim_kernel[grid](
        inp_flat, out_flat,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    out = out_flat.view_as(x_)
    if orig_dtype != torch.float32:
        out = out.to(orig_dtype)
    return out


class TritonSoftmaxLastDim(nn.Module):
    """
    Drop-in replacement for nn.Softmax(dim=-1) using the Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax_lastdim(x)


# ============================
# U-Net building blocks (Triton-accelerated softmax)
# ============================

class DoubleConvNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Keep high-performance PyTorch convolutions and batchnorm;
        # replace softmax with Triton implementation.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.softmax1 = TritonSoftmaxLastDim()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.softmax2 = TritonSoftmaxLastDim()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.softmax1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.softmax2(x)
        return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (will be doubled in each layer)
        """
        super(ModelNew, self).__init__()
        self.encoder1 = DoubleConvNew(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConvNew(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConvNew(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConvNew(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConvNew(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConvNew(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvNew(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvNew(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvNew(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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


# Example input helpers (can be used by external harness)
batch_size = 8
in_channels = 8
out_channels = 4
height = 64
width = 512
features = 64

def get_inputs():
    # For Triton, inputs should be moved to CUDA by the caller if needed.
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, features]
