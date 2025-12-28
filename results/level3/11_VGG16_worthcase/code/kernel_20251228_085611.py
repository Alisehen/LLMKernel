import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton kernels
# -----------------------------

@triton.jit
def conv3x3_bias_relu_kernel(
    x_ptr,            # float32* : [N, Cin, H, W]
    w_ptr,            # float32* : [K_total, Cout]  (K_total = Cin*9)
    bias_ptr,         # float32* : [Cout]
    y_ptr,            # float32* : [N, Cout, H, W]
    N, Cin, H, W, Cout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile in NHW dimension
    BLOCK_N: tl.constexpr,  # tile in Cout dimension
    BLOCK_K: tl.constexpr,  # K tile (Cin*3*3)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Flattened output spatial/batch index (M = N*H*W)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * H * W
    K_total = Cin * 9

    # Compute (n, h, w) from flattened offs_m
    hw_size = H * W
    n_idx = tl.floor_divide(offs_m, hw_size)
    hw = offs_m - n_idx * hw_size
    h_idx = tl.floor_divide(hw, W)
    w_idx = hw - h_idx * W

    # Broadcasted versions
    n_b = n_idx[:, None]
    h_b = h_idx[:, None]
    w_b = w_idx[:, None]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over K dimension (Cin*3*3)
    k0 = 0
    while k0 < K_total:
        k = k0 + offs_k
        mask_k = k < K_total

        # Map k -> (cin, kh, kw)
        cin_idx = tl.floor_divide(k, 9)
        rem = k - cin_idx * 9
        kh = tl.floor_divide(rem, 3)
        kw = rem - kh * 3

        cin_b = cin_idx[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Input coordinates with padding=1
        h_in = h_b + kh_b - 1
        w_in = w_b + kw_b - 1

        # Masks
        mask_m = offs_m < M
        mask_n = offs_n < Cout

        mask_in = (
            mask_m[:, None]
            & mask_k[None, :]
            & (n_b >= 0) & (n_b < N)
            & (cin_b >= 0) & (cin_b < Cin)
            & (h_in >= 0) & (h_in < H)
            & (w_in >= 0) & (w_in < W)
        )

        # Load input tile A: [BLOCK_M, BLOCK_K]
        a_ptrs = (
            x_ptr
            + n_b * stride_xn
            + cin_b * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        a = tl.load(a_ptrs, mask=mask_in, other=0.0)

        # Load weight tile B: [BLOCK_K, BLOCK_N]
        b_ptrs = (
            w_ptr
            + k[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        k0 += BLOCK_K

    # Add bias and ReLU
    mask_n = offs_n < Cout
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store to output [N, Cout, H, W]
    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < Cout)
    y_ptrs = (
        y_ptr
        + n_b * stride_yn
        + offs_n[None, :] * stride_yc
        + h_b * stride_yh
        + w_b * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_store)


@triton.jit
def gemm_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k = k0 + offs_k

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        mask_k = k < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)
        k0 += BLOCK_K

    # Add bias and ReLU
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k = k0 + offs_k

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        mask_k = k < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)
        k0 += BLOCK_K

    # Add bias (no activation)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = (
        c_ptr
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# -----------------------------
# Python wrappers
# -----------------------------

def conv3x3_bias_relu(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """
    Fused 3x3 conv (stride=1, padding=1) + bias + ReLU using Triton.
    x: [N, Cin, H, W], conv.weight: [Cout, Cin, 3, 3]
    """
    assert x.is_cuda
    weight = conv.weight
    bias = conv.bias
    assert weight.shape[2:] == (3, 3)
    assert conv.stride == (1, 1)
    assert conv.padding == (1, 1)
    assert conv.dilation == (1, 1)
    assert conv.groups == 1

    N, Cin, H, W = x.shape
    Cout = weight.shape[0]
    K_total = Cin * 9

    # Flatten weight to [K_total, Cout] contiguous for GEMM-friendly access
    w_mat = weight.view(Cout, K_total).transpose(0, 1).contiguous()

    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H * W, META["BLOCK_M"]),
        triton.cdiv(Cout, META["BLOCK_N"]),
    )

    conv3x3_bias_relu_kernel[grid](
        x, w_mat, bias, y,
        N, Cin, H, W, Cout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_mat.stride(0), w_mat.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


def linear_bias_relu(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """
    Fused Linear + bias + ReLU using Triton.
    x: [M, K], weight: [N, K] (PyTorch layout)
    """
    assert x.is_cuda
    weight = linear.weight
    bias = linear.bias
    M, K = x.shape
    N = weight.shape[0]

    # b_ptr expects [K, N] (input features x output features)
    b = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    gemm_bias_relu_kernel[grid](
        x, b, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return out


def linear_bias(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """
    Fused Linear + bias (no activation) using Triton.
    x: [M, K], weight: [N, K]
    """
    assert x.is_cuda
    weight = linear.weight
    bias = linear.bias
    M, K = x.shape
    N = weight.shape[0]

    b = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    gemm_bias_kernel[grid](
        x, b, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return out


# -----------------------------
# Optimized VGG16 Model
# -----------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        VGG16 with high-performance Triton kernels for:
          - All 3x3 Conv2d + ReLU layers
          - All Linear + ReLU layers
          - Final Linear layer
        MaxPool and Dropout remain as PyTorch ops.
        """
        super(ModelNew, self).__init__()

        # Reuse the exact module structure/parameters of the original model
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

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        Forward pass using Triton kernels for Conv+ReLU and Linear(+ReLU).
        x: [batch_size, 3, 224, 224]
        """
        # ---- Features (manual unrolling for fusion) ----
        # Block 1
        x = conv3x3_bias_relu(x, self.features[0])   # Conv 1_1 + ReLU
        x = conv3x3_bias_relu(x, self.features[2])   # Conv 1_2 + ReLU
        x = self.features[4](x)                      # MaxPool

        # Block 2
        x = conv3x3_bias_relu(x, self.features[5])   # Conv 2_1 + ReLU
        x = conv3x3_bias_relu(x, self.features[7])   # Conv 2_2 + ReLU
        x = self.features[9](x)                      # MaxPool

        # Block 3
        x = conv3x3_bias_relu(x, self.features[10])  # Conv 3_1 + ReLU
        x = conv3x3_bias_relu(x, self.features[12])  # Conv 3_2 + ReLU
        x = conv3x3_bias_relu(x, self.features[14])  # Conv 3_3 + ReLU
        x = self.features[16](x)                     # MaxPool

        # Block 4
        x = conv3x3_bias_relu(x, self.features[17])  # Conv 4_1 + ReLU
        x = conv3x3_bias_relu(x, self.features[19])  # Conv 4_2 + ReLU
        x = conv3x3_bias_relu(x, self.features[21])  # Conv 4_3 + ReLU
        x = self.features[23](x)                     # MaxPool

        # Block 5
        x = conv3x3_bias_relu(x, self.features[24])  # Conv 5_1 + ReLU
        x = conv3x3_bias_relu(x, self.features[26])  # Conv 5_2 + ReLU
        x = conv3x3_bias_relu(x, self.features[28])  # Conv 5_3 + ReLU
        x = self.features[30](x)                     # MaxPool

        # ---- Classifier ----
        x = torch.flatten(x, 1)                      # [B, 512*7*7]

        # Linear 1 + ReLU
        x = linear_bias_relu(x, self.classifier[0])
        x = self.classifier[2](x)                    # Dropout p=0

        # Linear 2 + ReLU
        x = linear_bias_relu(x, self.classifier[3])
        x = self.classifier[5](x)                    # Dropout p=0

        # Final Linear (no activation)
        x = linear_bias(x, self.classifier[6])
        return x
