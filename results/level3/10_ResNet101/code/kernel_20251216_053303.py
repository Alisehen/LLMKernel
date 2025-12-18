import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------
# Fused residual add + ReLU
# -------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def residual_add_relu_kernel(
    x_ptr,          # *f32
    identity_ptr,   # *f32
    y_ptr,          # *f32
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    identity = tl.load(identity_ptr + offs, mask=mask, other=0.0)

    out = x + identity
    out = tl.maximum(out, 0.0)

    tl.store(y_ptr + offs, out, mask=mask)


def residual_add_relu(x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """Fused residual add + ReLU: y = ReLU(x + identity)."""
    assert x.shape == identity.shape
    assert x.is_cuda and identity.is_cuda
    assert x.dtype == identity.dtype == torch.float32

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


# -------------------------------
# GEMM + bias for final linear
# -------------------------------
@triton.autotune(
    configs=[
        # Highly tuned for Ada (4090) under 99KB shared memory / block
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,  # 2 * 32KB = 64KB SRAM
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,  # ~96KB SRAM
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,  # ~96KB SRAM
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # *f32, [M, K]
    b_ptr,  # *f32, logically [K, N], physical strides passed in
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
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks along M and N are loop-invariant; compute once
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    c_mask = mask_m & mask_n

    # Pointers to first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop over full BLOCK_K tiles (no K-boundary masks)
    K_full = (K // BLOCK_K) * BLOCK_K
    k = 0
    while k < K_full:
        a = tl.load(a_ptrs, mask=mask_m, other=0.0)
        b = tl.load(b_ptrs, mask=mask_n, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Tail K tile (handles K not divisible by BLOCK_K)
    K_tail = K - K_full
    if K_tail > 0:
        k_mask_a = offs_k[None, :] < K_tail
        k_mask_b = offs_k[:, None] < K_tail

        a = tl.load(a_ptrs, mask=mask_m & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_n & k_mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

    # Fused bias add: same (offs_n, mask_n) as GEMM's N-dimension
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Linear layer using Triton: out = x @ weight.T + bias
    x: [M, K], weight: [N, K], bias: [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert bias.shape[0] == N

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    # Interpret weight as B with logical shape [K, N]
    # B[k, n] = weight[n, k] by using (stride_bk, stride_bn) = (stride along K, stride along N)
    linear_gemm_bias_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),  # stride_bk: increment in K
        weight.stride(0),  # stride_bn: increment in N
        out.stride(0),
        out.stride(1),
    )
    return out


# -------------------------------
# Model definition
# -------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
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

        # Final linear via Triton GEMM + bias
        if x.is_cuda and x.dtype == torch.float32:
            x = triton_linear(x, self.fc.weight, self.fc.bias)
        else:
            x = self.fc(x)

        return x
