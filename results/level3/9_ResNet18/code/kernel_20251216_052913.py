# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------------------
# 1. Triton Kernels
# ----------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def residual_add_relu_kernel(
    x_ptr,     # pointer to main branch tensor
    y_ptr,     # pointer to residual / identity tensor
    out_ptr,   # pointer to output tensor
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Fused residual add + ReLU, keep intermediates minimal
    x = x + y
    x = tl.maximum(x, 0.0)

    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32 for numeric stability, also optimal for tensor cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tl.dot will leverage tensor cores on Ada (TF32 / FP16 / BF16)
        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ----------------------------------------
# 2. Wrapper Functions
# ----------------------------------------

def fused_residual_add_relu(x: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """
    Fused elementwise: out = ReLU(x + identity)
    Both inputs are NCHW tensors of the same shape.
    """
    assert x.shape == identity.shape, "Shapes of main and identity must match"
    x_contig = x.contiguous()
    id_contig = identity.contiguous()
    out = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    residual_add_relu_kernel[grid](
        x_contig,
        id_contig,
        out,
        n_elements,
    )
    return out


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused linear: x @ weight.T + bias
    x:      [M, K]
    weight: [N, K] (nn.Linear.weight)
    bias:   [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA for Triton"
    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "Incompatible shapes for linear"
    assert bias.shape[0] == N, "Bias shape mismatch"

    # B is [K, N]
    B = weight.t().contiguous()
    A = x.contiguous()
    C = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_gemm_bias_kernel[grid](
        A,
        B,
        bias,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
    )
    return C


# ----------------------------------------
# 3. Model Definition Using Fused Kernels
# ----------------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

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

        x = fused_linear(x, self.fc.weight, self.fc.bias)

        return x


# BasicBlock definition kept outside ModelNew to match original structure
class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlockNew, self).__init__()
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

        out = fused_residual_add_relu(out, identity)
        return out
