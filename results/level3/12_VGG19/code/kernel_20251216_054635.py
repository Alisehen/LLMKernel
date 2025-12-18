import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Tuned for high-performance on Ada (RTX 4090) for large GEMMs
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=3,
        ),
        # Fallbacks for smaller / irregular shapes
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_gemm_bias_relu_kernel(
    x_ptr,  # (M, K)
    w_ptr,  # (K, N)  -- column-major in N: contiguous row-major tensor
    b_ptr,  # (N,)    -- optional, only used if HAS_BIAS=True
    y_ptr,  # (M, N)
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    RELU: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.bfloat16 / tl.float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Boundary masks for output tile (shared by all fused ops)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers for first K-tile
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulate in FP32 for numerical stability and to leverage TF32 / tensor cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop (reduction)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        x_mask = mask_m[:, None] & k_mask[None, :]
        w_mask = k_mask[:, None] & mask_n[None, :]

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matrix multiply fragment; allow_tf32 enables high-throughput tensor cores on FP32
        acc += tl.dot(x_block, w_block, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Fused bias add (broadcast along M) using SAME offs_n / mask_n
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

    # Fused ReLU activation on the accumulated result [M, N]
    if RELU:
        acc = tl.maximum(acc, 0.0)

    # Cast to requested output dtype after all math in FP32
    out = acc.to(OUT_DTYPE)

    # Store with a single, shared mask over [M, N]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, out, mask=y_mask)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, relu: bool):
    """
    High-performance fused Linear (GEMM + bias + ReLU) implemented in Triton.

    x      : (M, K)
    weight : (K, N)  -- NOTE: stored transposed compared to nn.Linear.weight
    bias   : (N,) or None
    return : (M, N)
    """
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
    assert x.dtype == weight.dtype
    if bias is not None:
        assert bias.dtype == x.dtype

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous() if bias is not None else None

    M, K = x_contig.shape
    K_w, N = w_contig.shape
    assert K_w == K, "Weight shape must be (in_features, out_features) == (K, N)"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Select Triton output dtype
    if x.dtype == torch.float16:
        out_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif x.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype for fused_linear: {x.dtype}")

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    linear_gemm_bias_relu_kernel[grid](
        x_contig, w_contig,
        b_contig if b_contig is not None else x_contig,  # dummy pointer if no bias
        y,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_contig.stride(0), w_contig.stride(1),
        y.stride(0), y.stride(1),
        RELU=relu,
        HAS_BIAS=(b_contig is not None),
        OUT_DTYPE=out_dtype,
    )
    return y


class _FusedLinearBase(nn.Module):
    """
    Linear layer whose weight is stored as (in_features, out_features) to
    maximize GEMM performance (row-major [K, N] for Triton matmul).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight layout: (K, N) == (in_features, out_features)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Match standard Linear initialization semantics explicitly
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.kaiming_uniform_(self.weight.T, a=math.sqrt(5))  # use transposed view for convenience
        if self.bias is not None:
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
        VGG19 with Triton-optimized fully-connected layers (Linear + bias + ReLU fused).
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
