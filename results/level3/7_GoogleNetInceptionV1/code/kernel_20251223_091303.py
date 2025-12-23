import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------
# Manually implemented elementwise activation helpers for Triton
# (not used in this model but provided per requirements)
# -----------------------------------------------------------

def _tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


def _tl_tanh(x):
    # tanh(x) = 2*sigmoid(2x) - 1
    return 2.0 * _tl_sigmoid(2.0 * x) - 1.0


def _tl_gelu(x):
    # Approximate GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    k = tl.sqrt(2.0 / 3.141592653589793)
    x3 = x * x * x
    inner = k * (x + 0.044715 * x3)
    return 0.5 * x * (1.0 + _tl_tanh(inner))


def _tl_silu(x):
    # SiLU(x) = x * sigmoid(x)
    return x * _tl_sigmoid(x)


def _tl_softmax(x, axis=-1):
    # Simple softmax along last axis (no explicit axis handling needed here)
    x_max = tl.max(x, axis=axis)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    exp_sum = tl.sum(exp_x, axis=axis)
    return exp_x / exp_sum


def _tl_mish(x):
    # Mish(x) = x * tanh(softplus(x)); softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    return x * _tl_tanh(softplus)


# Optionally expose them through `tl` namespace if desired
tl.sigmoid = _tl_sigmoid
tl.tanh = _tl_tanh
tl.gelu = _tl_gelu
tl.silu = _tl_silu
tl.softmax = _tl_softmax
tl.mish = _tl_mish


# -------------------------
# Triton Kernels & Wrappers
# -------------------------


@triton.jit
def relu_kernel(
    x_ptr,  # *f32
    y_ptr,  # *f32
    N,      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid covering the flattened output tensor
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    tl.store(y_ptr + offs, x, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Elementwise ReLU using Triton.
    Works on contiguous tensors of any shape/dtype supported by Triton.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    # Ensure contiguous memory layout for predictable indexing
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    N = x_contig.numel()
    if N == 0:
        return x_contig
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
    relu_kernel[grid](
        x_contig, y,
        N,
        BLOCK_SIZE=256,
        num_warps=4,
    )
    # Preserve original shape
    return y.view_as(x)


# -------------------------
# Fused GEMM + Bias kernel
# -------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8, num_stages=3
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4, num_stages=3
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4, num_stages=3
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr,       # *f32 [M, K]
    b_ptr,       # *f32 [K, N]  (weight^T)
    bias_ptr,    # *f32 [N]
    c_ptr,       # *f32 [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GEMM (A @ B) + bias add.

    Grid: 2D over (M, N).
    All fused ops (GEMM accumulation, bias add, store) share the SAME offs_m/offs_n
    and boundary masks derived from M and N.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets along the output matrix dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Boundary masks for output tile
    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Base pointers for A and B tiles
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        # Mask for A/B loads: combine output-dimension mask with K-range mask
        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        # Advance K-tile pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add: bias is [N], broadcast over M.
    # Use the same offs_n and mask_n that define the N dimension for the output tile.
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Store result, using the same offsets/mask tuple (offs_m, offs_n, out_mask)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=out_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fully-connected layer: y = x @ weight.T + bias

    x:      [M, K]
    weight: [N, K]
    bias:   [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors"
    assert x.dim() == 2, "Input to triton_linear must be 2D"
    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "Incompatible shapes for linear layer"

    # Prepare B as weight^T with shape [K, N]
    # This keeps GEMM layout simple and fast.
    b_mat = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0 or K == 0:
        return c

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    linear_kernel[grid](
        x, b_mat, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# -------------------------
# Model Definition
# -------------------------

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        # Keep concatenation in PyTorch to preserve branch separation
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Triton-optimized version of the given GoogLeNet-style model.
        Uses:
          - Triton ReLU for activation after conv1, conv2, conv3
          - Triton GEMM for the final fully-connected layer
        """
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)

        # Replace nn.Linear with explicit parameters so we can call Triton GEMM
        in_features = 1024
        self.fc_weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.fc_bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # conv1 + ReLU (Triton) + maxpool1
        x = self.conv1(x)
        x = triton_relu(x)
        x = self.maxpool1(x)

        # conv2 + ReLU (Triton)
        x = self.conv2(x)
        x = triton_relu(x)

        # conv3 + ReLU (Triton) + maxpool2
        x = self.conv3(x)
        x = triton_relu(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # Final FC with Triton GEMM (fused matmul + bias)
        x = triton_linear(x, self.fc_weight, self.fc_bias)

        return x
