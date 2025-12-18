import torch
import torch.nn as nn
import triton
import triton.language as tl


# =========================
# Triton Kernels
# =========================

@triton.autotune(
    configs=[
        # Smaller tiles to keep registers per thread low while still efficient
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['N', 'H', 'W', 'C_in', 'C_out'],
)
@triton.jit
def conv3x3_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over (N * H * W)
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # tile over C_in * 3 * 3 (reduction)
):
    """
    Fused 3x3 Conv2d (stride=1, padding=1, dilation=1, groups=1) + bias + ReLU
    implemented as implicit GEMM:
        [N*H*W, C_in*3*3] @ [C_in*3*3, C_out] -> [N*H*W, C_out]
    Layout:
      x: [N, C_in, H, W]      (NCHW)
      w: [C_out, C_in, 3, 3]  (OIHW)
      y: [N, C_out, H, W]     (NCHW)
    """
    pid_m = tl.program_id(0)  # along N*H*W
    pid_n = tl.program_id(1)  # along C_out

    P = N * H * W
    K_total = C_in * 9  # C_in * 3 * 3

    # Offsets for output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Map flattened spatial index -> (n, h, w)
    tmp = offs_m
    w_idx = tmp % W
    tmp = tmp // W
    h_idx = tmp % H
    n_idx = tmp // H

    n = n_idx[:, None]  # [BM, 1]
    h = h_idx[:, None]  # [BM, 1]
    w = w_idx[:, None]  # [BM, 1]
    co = offs_n[None, :]  # [1, BN]

    # Hints for compiler (alignment / div)
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K_total:
        k_idx = k + offs_k  # [BK]
        mask_k = k_idx < K_total

        # Map k_idx -> (ci, kh, kw)
        ci_k = k_idx // 9          # [BK]
        rem_k = k_idx % 9
        kh_k = rem_k // 3          # [BK]
        kw_k = rem_k % 3           # [BK]

        # Broadcast for pointer arithmetic
        ci = ci_k[None, :]         # [1, BK]
        kh = kh_k[None, :]         # [1, BK]
        kw = kw_k[None, :]         # [1, BK]

        # Input coordinates for each (m, k)
        ih = h + kh - 1            # [BM, BK]
        iw = w + kw - 1            # [BM, BK]

        # In-bounds mask for input
        in_h = (ih >= 0) & (ih < H)
        in_w = (iw >= 0) & (iw < W)
        in_bounds = in_h & in_w

        # Input load
        x_ptrs = (
            x_ptr
            + n * stride_xn
            + ci * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        x = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & in_bounds & mask_k[None, :],
            other=0.0,
        )

        # Weight load
        ci_w = ci_k[:, None]  # [BK, 1]
        kh_w = kh_k[:, None]  # [BK, 1]
        kw_w = kw_k[:, None]  # [BK, 1]
        w_ptrs = (
            w_ptr
            + co * stride_wo
            + ci_w * stride_wc
            + kh_w * stride_wh
            + kw_w * stride_ww
        )
        w_tile = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # GEMM accumulate (TF32 to hit tensor cores)
        acc += tl.dot(x, w_tile, allow_tf32=True)

        k += BLOCK_K

    # Fused bias add (cheap; applied after reduction)
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    acc += bias[None, :]

    # Fused ReLU
    acc = tl.maximum(acc, 0.0)

    # Store output y[n, co, h, w]
    y_ptrs = (
        y_ptr
        + n * stride_yn
        + co * stride_yc
        + h * stride_yh
        + w * stride_yw
    )
    # Recompute simple mask to avoid keeping a large mask tensor live
    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along M
    BLOCK_N: tl.constexpr,  # tile along N
    BLOCK_K: tl.constexpr,  # reduction tile
):
    """
    Fused Linear (GEMM) + bias (+ optional ReLU).

    Computes: C = A @ B + bias, where
      A: [M, K]
      B: [K, N]  (weight^T)
      bias: [N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    mask_m = offs_m < M
    mask_n = offs_n < N

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Initial pointers
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    k = 0
    while k < K:
        k_idx = k + offs_k
        mask_k = k_idx < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Optional fused ReLU
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output, recomputing mask to avoid persisting a large mask tensor
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


# =========================
# Wrapper Functions
# =========================

def conv3x3_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [N, C_in, H, W]
    weight: [C_out, C_in, 3, 3]
    bias:   [C_out]
    returns [N, C_out, H, W]
    """
    assert x.dim() == 4
    assert weight.dim() == 4 and weight.shape[2] == 3 and weight.shape[3] == 3
    assert bias.dim() == 1 and bias.shape[0] == weight.shape[0]
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(N * H * W, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv3x3_relu_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y


def linear_bias_triton(
    x: torch.Tensor,      # [M, K]
    weight: torch.Tensor, # [N, K] (PyTorch Linear: out_features, in_features)
    bias: torch.Tensor,   # [N]
    relu: bool,
) -> torch.Tensor:
    """
    Linear layer using Triton GEMM:
        out = x @ weight.T + bias
        optionally followed by ReLU.
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    assert bias.dim() == 1
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert bias.shape[0] == N

    # B is [K, N] (weight^T) for GEMM-friendly layout
    b = weight.t().contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    linear_bias_kernel[grid](
        x, b, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        HAS_RELU=relu,
    )

    return out


# =========================
# Optimized VGG16-like Model
# =========================

class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        """
        VGG16-like model using fused Triton kernels for:
          - Conv2d (3x3, stride=1, padding=1, bias=True) + ReLU
          - Linear + bias (+ optional ReLU)
        We use nn.Conv2d / nn.Linear only as parameter containers and run
        all heavy compute with Triton.
        """
        super().__init__()

        # Convolution blocks (parameters only; we will call Triton conv)
        # Block 1
        self.conv1_1 = nn.Conv2d(3,   64, kernel_size=3, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64,  64, kernel_size=3, padding=1, bias=True)

        # Block 2
        self.conv2_1 = nn.Conv2d(64,  128, kernel_size=3, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)

        # MaxPool layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier (parameters only; compute with Triton)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 4096, bias=True)
        self.fc3 = nn.Linear(4096, num_classes, bias=True)

        self.dropout1 = nn.Dropout(p=0.0)
        self.dropout2 = nn.Dropout(p=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, 3, 224, 224]
        returns: [batch_size, num_classes]
        """
        assert x.is_cuda, "ModelNew expects CUDA tensors for Triton kernels"
        assert x.dtype == torch.float32, "Kernels are specialized for float32"

        # Block 1
        x = conv3x3_relu_triton(x, self.conv1_1.weight, self.conv1_1.bias)
        x = conv3x3_relu_triton(x, self.conv1_2.weight, self.conv1_2.bias)
        x = self.pool1(x)

        # Block 2
        x = conv3x3_relu_triton(x, self.conv2_1.weight, self.conv2_1.bias)
        x = conv3x3_relu_triton(x, self.conv2_2.weight, self.conv2_2.bias)
        x = self.pool2(x)

        # Block 3
        x = conv3x3_relu_triton(x, self.conv3_1.weight, self.conv3_1.bias)
        x = conv3x3_relu_triton(x, self.conv3_2.weight, self.conv3_2.bias)
        x = conv3x3_relu_triton(x, self.conv3_3.weight, self.conv3_3.bias)
        x = self.pool3(x)

        # Block 4
        x = conv3x3_relu_triton(x, self.conv4_1.weight, self.conv4_1.bias)
        x = conv3x3_relu_triton(x, self.conv4_2.weight, self.conv4_2.bias)
        x = conv3x3_relu_triton(x, self.conv4_3.weight, self.conv4_3.bias)
        x = self.pool4(x)

        # Block 5
        x = conv3x3_relu_triton(x, self.conv5_1.weight, self.conv5_1.bias)
        x = conv3x3_relu_triton(x, self.conv5_2.weight, self.conv5_2.bias)
        x = conv3x3_relu_triton(x, self.conv5_3.weight, self.conv5_3.bias)
        x = self.pool5(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier with fused Triton GEMM + bias + ReLU
        x = linear_bias_triton(x, self.fc1.weight, self.fc1.bias, relu=True)
        x = self.dropout1(x)
        x = linear_bias_triton(x, self.fc2.weight, self.fc2.bias, relu=True)
        x = self.dropout2(x)
        x = linear_bias_triton(x, self.fc3.weight, self.fc3.bias, relu=False)

        return x
