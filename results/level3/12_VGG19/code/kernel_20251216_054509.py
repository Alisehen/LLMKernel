import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_gemm_bias_relu_kernel(
    x_ptr,  # (M, K)
    w_ptr,  # (N, K)
    b_ptr,  # (N,)
    y_ptr,  # (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_ym, stride_yn,
    relu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Treat w as (K, N) = weight.T, where original w is (N, K)
    w_ptrs = w_ptr + offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < k_remaining)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_block, w_block, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if relu:
        acc = tl.maximum(acc, 0.0)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, relu: bool):
    """
    x: (M, K)
    weight: (N, K)  -- same as nn.Linear.weight
    bias: (N,)
    returns: (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    M, K = x_contig.shape
    N = w_contig.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    linear_gemm_bias_relu_kernel[grid](
        x_contig, w_contig, b_contig, y,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_contig.stride(0), w_contig.stride(1),
        y.stride(0), y.stride(1),
        relu=relu,
    )
    return y


class _FusedLinearBase(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


class FusedLinearReLU(_FusedLinearBase):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear(x, self.weight, self.bias, relu=True)


class FusedLinear(_FusedLinearBase):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear(x, self.weight, self.bias, relu=False)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        VGG19 with Triton-optimized fully-connected layers (Linear + ReLU fused).
        """
        super(ModelNew, self).__init__()

        # Convolutional feature extractor: same as original VGG19 definition
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
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier with fused Linear+ReLU implemented in Triton
        self.classifier = nn.Sequential(
            FusedLinearReLU(512 * 7 * 7, 4096),
            nn.Dropout(p=0.0),
            FusedLinearReLU(4096, 4096),
            nn.Dropout(p=0.0),
            FusedLinear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 3, 224, 224)
        returns: (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
