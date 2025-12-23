import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_bias_relu_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (weight^T)
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        k_mask_a = (offs_k[None, :] + k) < K
        k_mask_b = (offs_k[:, None] + k) < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask_a,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & k_mask_b,
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (weight^T)
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        k_mask_a = (offs_k[None, :] + k) < K
        k_mask_b = (offs_k[:, None] + k) < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask_a,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & k_mask_b,
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + Bias + ReLU using Triton.
    x: [M, K], weight: [N, K], bias: [N]
    Returns: [M, N]
    """
    # Fallback for non-CUDA tensors
    if not x.is_cuda:
        return torch.relu(torch.nn.functional.linear(x, weight, bias))

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]

    # weight is [N, K] -> b is [K, N]
    b = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_relu_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return c


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused Linear + Bias using Triton.
    x: [M, K], weight: [N, K], bias: [N]
    Returns: [M, N]
    """
    if not x.is_cuda:
        return torch.nn.functional.linear(x, weight, bias)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]

    b = weight.t().contiguous()
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        """
        VGG16 model with Triton-optimized classifier (Linear+Bias(+ReLU) fused).
        """
        super(ModelNew, self).__init__()

        # Convolutional feature extractor: keep standard highly-optimized PyTorch Conv2d
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Triton-optimized classifier parameters
        in_features = 512 * 7 * 7
        self.fc1_weight = nn.Parameter(torch.empty(4096, in_features))
        self.fc1_bias = nn.Parameter(torch.empty(4096))

        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096))
        self.fc2_bias = nn.Parameter(torch.empty(4096))

        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 4096))
        self.fc3_bias = nn.Parameter(torch.empty(num_classes))

        # Initialize weights similar to nn.Linear default (Kaiming-uniform)
        for weight, bias in [
            (self.fc1_weight, self.fc1_bias),
            (self.fc2_weight, self.fc2_bias),
            (self.fc3_weight, self.fc3_bias),
        ]:
            fan_in = weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VGG16 model with Triton-optimized classifier.
        x: [batch_size, 3, 224, 224]
        returns: [batch_size, num_classes]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 512*7*7]

        # Triton fused linear + bias + ReLU
        x = triton_linear_relu(x, self.fc1_weight, self.fc1_bias)
        x = triton_linear_relu(x, self.fc2_weight, self.fc2_bias)

        # Final Triton linear + bias (no activation)
        x = triton_linear(x, self.fc3_weight, self.fc3_bias)
        return x
