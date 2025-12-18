# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Fused residual add + ReLU (elementwise)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Smaller tile => lower register pressure, good fallback
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        # Larger tile for higher throughput if registers allow
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def residual_add_relu_kernel(
    x_ptr,          # *f32
    identity_ptr,   # *f32
    y_ptr,          # *f32
    n_elements,     # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    identity = tl.load(identity_ptr + offs, mask=mask, other=0.0)

    out = x + identity
    out = tl.maximum(out, 0.0)  # ReLU

    tl.store(y_ptr + offs, out, mask=mask)


def residual_add_relu(x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """
    Fused residual addition + ReLU using Triton.
    Inputs: x, identity: same shape, float32 CUDA tensors.
    """
    assert x.shape == identity.shape, "x and identity must have the same shape"
    assert x.is_cuda and identity.is_cuda, "Inputs must be on CUDA device"
    assert x.dtype == identity.dtype == torch.float32, "Kernel currently supports float32 only"

    y = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    residual_add_relu_kernel[grid](
        x.view(-1),
        identity.view(-1),
        y.view(-1),
        n_elements,
    )
    return y


# -----------------------------------------------------------------------------
# GEMM + Bias (final linear layer)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Balanced tile, good starting point
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Lower register pressure on M dimension
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
        # Lower register pressure on N dimension (good fallback)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # *f32, [M, K]
    b_ptr,  # *f32, [K, N] (weight^T)
    bias_ptr,  # *f32, [N]
    c_ptr,  # *f32, [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling of output matrix C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for rows/cols of the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    # Loop over K dimension
    while k < K:
        k_offsets = k + offs_k

        # Masks for valid K indices
        k_mask_row = k_offsets[None, :] < K    # shape [1, BLOCK_K]
        k_mask_col = k_offsets[:, None] < K    # shape [BLOCK_K, 1]

        a_mask = (offs_m[:, None] < M) & k_mask_row
        b_mask = k_mask_col & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Use Tensor Cores via TF32 when possible
        acc += tl.dot(a, b, allow_tf32=True)

        # Move pointers to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Linear layer implemented as GEMM + bias using Triton.
    x: [M, K], weight: [N, K], bias: [N] (all float32, CUDA)
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be on CUDA device"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "Kernel currently supports float32 only"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "Incompatible shapes for linear layer"
    assert bias.shape[0] == N, "Bias shape mismatch"

    # Convert weight to [K, N] for GEMM (column-major w.r.t N)
    b = weight.t().contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_gemm_bias_kernel[grid](
        x,
        b,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    return out


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels * expansion, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused residual add + ReLU via Triton
        if out.is_cuda and identity.is_cuda and out.dtype == torch.float32:
            out = residual_add_relu(out, identity)
        else:
            out = self.relu(out + identity)

        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        """
        :param layers: List of integers specifying the number of blocks in each layer
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Parameters live in nn.Linear; Triton used in forward
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Use Triton for the final linear layer when possible
        if x.is_cuda and x.dtype == torch.float32:
            x = triton_linear(x, self.fc.weight, self.fc.bias)
        else:
            x = self.fc(x)

        return x
