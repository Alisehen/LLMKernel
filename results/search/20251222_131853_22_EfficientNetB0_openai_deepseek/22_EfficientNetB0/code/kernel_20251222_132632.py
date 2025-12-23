# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# =======================
# Helper activation ops (for potential future fusion)
# =======================

def tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


def tl_tanh(x):
    e2x = tl.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


def tl_gelu(x):
    # Approximate GELU (tanh version)
    # 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    x3 = x * x * x
    inner = k0 * (x + k1 * x3)
    return 0.5 * x * (1.0 + tl_tanh(inner))


def tl_silu(x):
    return x * tl_sigmoid(x)


def tl_mish(x):
    # x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    return x * tl_tanh(softplus)


def tl_softmax(x, axis=-1):
    # Simple, not used here; included for completeness
    x_max = tl.max(x, axis=axis)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=axis)
    return x_exp / x_sum


# =======================
# Triton MatMul + Bias (Autotuned, 2D grid)
# =======================

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 256,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id for output tile C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Create 2D coordinates for output C
    offs_m_broadcast = offs_m[:, None]                # [BLOCK_M, 1]
    offs_n_broadcast = offs_n[None, :]                # [1, BLOCK_N]

    # Shared boundary mask for all operations on output tile
    c_mask = (offs_m_broadcast < M) & (offs_n_broadcast < N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)            # [BLOCK_K]

        # Pointers for A and B tiles
        a_ptrs = a_ptr + offs_m_broadcast * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n_broadcast * stride_bn

        # Masks for A and B (same grid, extended with K dimension)
        a_mask = (offs_m_broadcast < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n_broadcast < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Compute partial matmul for this K-slice (TF32 allowed on 4090)
        acc += tl.dot(a, b, allow_tf32=True)

    # Fused bias add: bias is [N], broadcast over M dimension.
    # Uses same output-indexing (offs_n) and boundary condition on N.
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]  # broadcast over M

    # Store result back to C
    c_ptrs = c_ptr + offs_m_broadcast * stride_cm + offs_n_broadcast * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_bias(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance matmul with fused bias using Triton.

    Computes: out = a @ w.T + bias
    Shapes:
        a: [M, K]
        w: [N, K]  (like nn.Linear weight: [out_features, in_features])
        bias: [N]
        out: [M, N]
    """
    assert a.is_cuda and w.is_cuda and bias.is_cuda
    assert a.dtype == w.dtype == bias.dtype == torch.float32

    M, K = a.shape
    N = w.shape[0]
    assert w.shape[1] == K
    assert bias.shape[0] == N

    # B is logically [K, N] where B[k, n] == w[n, k]
    b_ptr = w
    stride_bk = w.stride(1)
    stride_bn = w.stride(0)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    matmul_bias_kernel[grid](
        a, b_ptr, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        stride_bk, stride_bn,
        c.stride(0), c.stride(1),
    )
    return c


# =======================
# 1x1 Conv via MatMul
# =======================

def triton_conv1x1_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    1x1 convolution implemented as matmul over flattened spatial positions.

    x:      [N, C_in, H, W]  (NCHW, contiguous)
    weight: [C_out, C_in, 1, 1]
    bias:   [C_out]

    Returns:
        y: [N, C_out, H, W] (NCHW, contiguous)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape[1] == C_in
    assert weight.shape[2] == 1 and weight.shape[3] == 1
    assert bias.shape[0] == C_out

    # Flatten spatial positions, move channels to last dimension:
    # A[m, k] = x[n, k, h, w] with m = n*H*W + h*W + w
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(-1, C_in)   # [M, C_in]
    w_2d = weight.view(C_out, C_in).contiguous()               # [C_out, C_in]

    # GEMM: [M, C_in] x [C_out, C_in]^T -> [M, C_out]
    out_2d = triton_matmul_bias(x_2d, w_2d, bias)              # [M, C_out]

    # Reshape back to [N, C_out, H, W]
    y = out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    return y


# =======================
# High-performance Layers
# =======================

class LinearTriton(nn.Module):
    """
    Drop-in replacement for nn.Linear using Triton GEMM (matmul + bias).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Match nn.Linear parameter shapes: [out_features, in_features]
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_buf", None, persistent=False)
        else:
            self.bias = None
            # Persistent zero-bias buffer to avoid allocations in forward
            self.register_buffer("bias_buf", torch.zeros(out_features), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Same init as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "LinearTriton only supports CUDA tensors"
        x = x.contiguous()
        if self.bias is not None:
            bias_vec = self.bias
        else:
            # Ensure bias_buf is on the right device / dtype
            if self.bias_buf.device != x.device or self.bias_buf.dtype != x.dtype:
                self.bias_buf = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
            bias_vec = self.bias_buf
        return triton_matmul_bias(x, self.weight, bias_vec)


class Conv1x1Triton(nn.Module):
    """
    1x1 convolution (no padding, stride=1, groups=1) implemented with Triton GEMM.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # [out_channels, in_channels, 1, 1]
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            self.register_buffer("bias_buf", None, persistent=False)
        else:
            self.bias = None
            self.register_buffer("bias_buf", torch.zeros(out_channels), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Follow nn.Conv2d Kaiming initializer for conv weights
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Conv1x1Triton only supports CUDA tensors"

        if self.bias is not None:
            bias_vec = self.bias
        else:
            # Ensure bias_buf is on the right device / dtype
            if self.bias_buf.device != x.device or self.bias_buf.dtype != x.dtype:
                self.bias_buf = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            bias_vec = self.bias_buf

        return triton_conv1x1_nchw(x, self.weight, bias_vec)


# =======================
# MBConv Block (with 1x1 convs accelerated)
# =======================

class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                Conv1x1Triton(in_channels, hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            Conv1x1Triton(hidden_dim, out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x

        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        if self.use_residual:
            x = x + identity

        return x


# =======================
# EfficientNetB0 (ModelNew)
# =======================

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Initial 3x3 convolution
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)

        # MBConv blocks with Triton-accelerated 1x1 convolutions
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6),
        )

        # Final pointwise convolution (Triton-accelerated)
        self.conv2 = Conv1x1Triton(320, 1280, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # Fully connected layer (Triton-accelerated)
        self.fc = LinearTriton(1280, num_classes, bias=True)

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
