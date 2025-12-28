# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


# -----------------------------
# Triton Kernels
# -----------------------------

@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # tile in M
    BLOCK_N: tl.constexpr,  # tile in N
    BLOCK_K: tl.constexpr,  # tile in K
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.float32),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def add_relu_kernel(
    a_ptr,  # input A
    b_ptr,  # input B
    c_ptr,  # output
    NUMEL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUMEL

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    out = tl.maximum(out, 0.0)
    tl.store(c_ptr + offs, out, mask=mask)


@triton.jit
def global_avgpool2d_kernel(
    x_ptr,   # [N, C, H, W]
    y_ptr,   # [N, C]
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    stride_on, stride_oc,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    nc_mask = offs_nc < (N * C)

    n = offs_nc // C
    c = offs_nc % C

    S = H * W
    acc = tl.zeros((BLOCK_NC,), dtype=tl.float32)

    for s in range(0, S, BLOCK_HW):
        offs_s = s + tl.arange(0, BLOCK_HW)
        s_mask = offs_s < S

        h = offs_s // W
        w = offs_s % W

        ptrs = (
            x_ptr
            + n[:, None] * stride_n
            + c[:, None] * stride_c
            + h[None, :] * stride_h
            + w[None, :] * stride_w
        )
        vals = tl.load(
            ptrs,
            mask=nc_mask[:, None] & s_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(vals, axis=1)

    norm = 1.0 / (H * W)
    acc = acc * norm

    y_ptrs = y_ptr + n * stride_on + c * stride_oc
    tl.store(y_ptrs, acc, mask=nc_mask)


# -----------------------------
# Python Wrappers
# -----------------------------

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [N, K] (PyTorch Linear weight is [N, K])
    bias: [N]
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    M, K = x.shape
    N = weight.shape[0]
    # Convert to [K, N] for better GEMM layout
    b = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    linear_gemm_bias_kernel[grid](
        x, b, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return y


def triton_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise (a + b).relu(), fused.
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    c = torch.empty_like(a)
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    c_flat = c.view(-1)
    numel = a_flat.numel()

    grid = lambda META: (triton.cdiv(numel, META["BLOCK_SIZE"]),)
    add_relu_kernel[grid](
        a_flat, b_flat, c_flat,
        numel,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    return c


def triton_global_avgpool2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, H, W] -> [N, C] (mean over H,W)
    """
    assert x.is_cuda
    assert x.ndim == 4
    N, C, H, W = x.shape

    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(N * C, META["BLOCK_NC"]),)
    x_contig = x.contiguous()
    global_avgpool2d_kernel[grid](
        x_contig, y,
        N, C, H, W,
        x_contig.stride(0), x_contig.stride(1),
        x_contig.stride(2), x_contig.stride(3),
        y.stride(0), y.stride(1),
        BLOCK_NC=64,
        BLOCK_HW=64,
        num_warps=4,
        num_stages=2,
    )
    # Return as [N, C, 1, 1] to match AdaptiveAvgPool2d((1,1))
    return y.view(N, C, 1, 1)


# -----------------------------
# Model Definition
# -----------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused residual add + ReLU via Triton
        out = triton_add_relu(out, identity)

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Keep for structure, but we'll use Triton global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Use nn.Linear for parameter management; forward uses Triton kernel
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

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
        # x: [B, 3, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Triton global average pooling instead of PyTorch AdaptiveAvgPool2d
        x = triton_global_avgpool2d(x)  # [B, 512, 1, 1]

        x = torch.flatten(x, 1)  # [B, 512]

        # Triton linear instead of nn.Linear forward
        x = triton_linear(x, self.fc.weight, self.fc.bias)

        return x
