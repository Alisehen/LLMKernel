# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton kernels
# -----------------------------

@triton.jit
def conv3x3_bias_relu_kernel(
    x_ptr,            # float* : [N, Cin, H, W]
    w_ptr,            # float* : [Cout, Cin, 3, 3]
    bias_ptr,         # float* : [Cout]
    y_ptr,            # float* : [N, Cout, H, W]
    N, Cin, H, W, Cout,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,  # block size in Cout dimension
):
    # Each program instance computes one output pixel (n, h, w)
    # for a block of output channels [co, co + BLOCK_CO)
    pid_nhw = tl.program_id(0)
    pid_co_block = tl.program_id(1)

    # Decode pid_nhw -> (n, h, w)
    HW = H * W
    n = pid_nhw // HW
    hw = pid_nhw - n * HW
    h = hw // W
    w = hw - h * W

    # Offsets in output-channel dimension
    offs_co = pid_co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < Cout

    # Accumulator for this (n, h, w) over BLOCK_CO output channels
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels
    c = 0
    while c < Cin:
        # Base pointer for x[n, c, :, :]
        x_nc_base = x_ptr + n * stride_xn + c * stride_xc

        # 3x3 convolution window with padding=1
        # kh, kw are Python constants -> fully unrolled by Triton
        # We load one scalar x_val and a vector of weights per (kh, kw)
        # and accumulate acc += w_val * x_val
        # kh = 0
        h_in = h + 0 - 1
        if (True):
            w_in = w + 0 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_00 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_00 = tl.load(x_ptr_00, mask=mask_in, other=0.0)
            w_ptrs_00 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 0 * stride_wh
                + 0 * stride_ww
            )
            w_val_00 = tl.load(w_ptrs_00, mask=mask_co, other=0.0)
            acc += w_val_00 * x_val_00

            w_in = w + 1 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_01 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_01 = tl.load(x_ptr_01, mask=mask_in, other=0.0)
            w_ptrs_01 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 0 * stride_wh
                + 1 * stride_ww
            )
            w_val_01 = tl.load(w_ptrs_01, mask=mask_co, other=0.0)
            acc += w_val_01 * x_val_01

            w_in = w + 2 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_02 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_02 = tl.load(x_ptr_02, mask=mask_in, other=0.0)
            w_ptrs_02 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 0 * stride_wh
                + 2 * stride_ww
            )
            w_val_02 = tl.load(w_ptrs_02, mask=mask_co, other=0.0)
            acc += w_val_02 * x_val_02

        # kh = 1
        h_in = h + 1 - 1
        if (True):
            w_in = w + 0 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_10 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_10 = tl.load(x_ptr_10, mask=mask_in, other=0.0)
            w_ptrs_10 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 1 * stride_wh
                + 0 * stride_ww
            )
            w_val_10 = tl.load(w_ptrs_10, mask=mask_co, other=0.0)
            acc += w_val_10 * x_val_10

            w_in = w + 1 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_11 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_11 = tl.load(x_ptr_11, mask=mask_in, other=0.0)
            w_ptrs_11 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 1 * stride_wh
                + 1 * stride_ww
            )
            w_val_11 = tl.load(w_ptrs_11, mask=mask_co, other=0.0)
            acc += w_val_11 * x_val_11

            w_in = w + 2 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_12 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_12 = tl.load(x_ptr_12, mask=mask_in, other=0.0)
            w_ptrs_12 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 1 * stride_wh
                + 2 * stride_ww
            )
            w_val_12 = tl.load(w_ptrs_12, mask=mask_co, other=0.0)
            acc += w_val_12 * x_val_12

        # kh = 2
        h_in = h + 2 - 1
        if (True):
            w_in = w + 0 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_20 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_20 = tl.load(x_ptr_20, mask=mask_in, other=0.0)
            w_ptrs_20 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 2 * stride_wh
                + 0 * stride_ww
            )
            w_val_20 = tl.load(w_ptrs_20, mask=mask_co, other=0.0)
            acc += w_val_20 * x_val_20

            w_in = w + 1 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_21 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_21 = tl.load(x_ptr_21, mask=mask_in, other=0.0)
            w_ptrs_21 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 2 * stride_wh
                + 1 * stride_ww
            )
            w_val_21 = tl.load(w_ptrs_21, mask=mask_co, other=0.0)
            acc += w_val_21 * x_val_21

            w_in = w + 2 - 1
            mask_in = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
            x_ptr_22 = x_nc_base + h_in * stride_xh + w_in * stride_xw
            x_val_22 = tl.load(x_ptr_22, mask=mask_in, other=0.0)
            w_ptrs_22 = (
                w_ptr
                + offs_co * stride_wn
                + c * stride_wc
                + 2 * stride_wh
                + 2 * stride_ww
            )
            w_val_22 = tl.load(w_ptrs_22, mask=mask_co, other=0.0)
            acc += w_val_22 * x_val_22

        c += 1

    # Add bias and ReLU
    bias = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)
    acc = acc + bias
    acc = tl.maximum(acc, 0.0)

    # Store output y[n, co, h, w]
    y_ptrs = (
        y_ptr
        + n * stride_yn
        + offs_co * stride_yc
        + h * stride_yh
        + w * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_co)


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
# Python wrappers / helpers
# -----------------------------

def conv3x3_bias_relu(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    """
    Fused 3x3 conv (stride=1, padding=1) + bias + ReLU using Triton.
    x: [N, Cin, H, W], conv.weight: [Cout, Cin, 3, 3]
    """
    assert x.is_cuda
    assert conv.weight.shape[2:] == (3, 3)
    assert conv.stride == (1, 1)
    assert conv.padding == (1, 1)
    assert conv.dilation == (1, 1)
    assert conv.groups == 1
    assert conv.bias is not None

    N, Cin, H, W = x.shape
    Cout = conv.weight.shape[0]

    y = torch.empty((N, Cout, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        max(1, N * H * W),                       # each program handles one (n, h, w)
        triton.cdiv(Cout, META["BLOCK_CO"]),     # blocks over output channels
    )

    conv3x3_bias_relu_kernel[grid](
        x, conv.weight, conv.bias, y,
        N, Cin, H, W, Cout,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv.weight.stride(0), conv.weight.stride(1),
        conv.weight.stride(2), conv.weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_CO=64,
        num_warps=4,
        num_stages=2,
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

        # Same architecture as the reference VGG16
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
        Returns: [batch_size, num_classes]
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
