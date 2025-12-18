# Complete ModelNew code with optimized Triton kernels

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def conv2d_relu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, H, W, H_out, W_out,
    stride_h, stride_w,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    Cin: tl.constexpr, Cout: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # Each program computes all output channels for one (b, ho, wo) location,
    # vectorized over channels (BLOCK_CO).
    pid = tl.program_id(0)

    n_hw = H_out * W_out
    b = pid // n_hw
    hw = pid % n_hw
    ho = hw // W_out
    wo = hw % W_out

    offs_co = tl.arange(0, BLOCK_CO)
    mask_co = offs_co < Cout

    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Unrolled over Cin, Kh, Kw (all constexpr)
    for ci in range(Cin):
        for kh in range(Kh):
            ih = ho * stride_h + kh
            for kw in range(Kw):
                iw = wo * stride_w + kw

                x_ptr_scalar = x_ptr + (
                    b * stride_xb
                    + ci * stride_xc
                    + ih * stride_xh
                    + iw * stride_xw
                )
                x_val = tl.load(x_ptr_scalar)

                w_ptrs = w_ptr + (
                    offs_co * stride_wco
                    + ci * stride_wci
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_val = tl.load(w_ptrs, mask=mask_co, other=0.0)
                acc += x_val * w_val

    # Add bias
    bias = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)
    acc += bias

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store
    y_ptrs = y_ptr + (
        b * stride_yb
        + offs_co * stride_yc
        + ho * stride_yh
        + wo * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_co)


@triton.jit
def linear_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        k_remaining = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        k_remaining = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias only (no activation)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[:, None] < N),
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def conv2d_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    NCHW conv2d with kernel=5x5, stride=1, no padding, fused with bias and ReLU.
    Uses a simple per-(b, h, w) kernel, vectorized over output channels.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    weight = weight.contiguous()

    B, Cin, H, W = x.shape
    Cout, Cin_w, Kh, Kw = weight.shape
    assert Cin == Cin_w
    assert Kh == Kw  # here always 5x5
    stride_h = 1
    stride_w = 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1

    y = torch.empty((B, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = (B * H_out * W_out,)

    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        B, H, W, H_out, W_out,
        stride_h, stride_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        Cin=Cin,
        Cout=Cout,
        Kh=Kh,
        Kw=Kw,
        BLOCK_CO=16,  # power of 2, >= max Cout (16)
    )

    return y


def linear_bias_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused x @ weight.T + bias with ReLU.
    x: (M, K)
    weight: (N, K) (PyTorch-style)  -> we internally use weight.T of shape (K, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    # weight is (out_features, in_features); we use transposed (K, N)
    w_t = weight.t().contiguous()

    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    linear_bias_relu_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y


def linear_bias_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused x @ weight.T + bias (no activation).
    x: (M, K)
    weight: (N, K) (PyTorch-style) -> internally weight.T (K, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    w_t = weight.t().contiguous()

    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    linear_bias_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    LeNet-5-style model with Triton-accelerated conv and linear layers.

    - Conv layers: fused Conv2d + Bias + ReLU via Triton.
    - FC layers: fused Linear + Bias (+ ReLU where needed) via Triton.
    - MaxPool remains in PyTorch (F.max_pool2d).
    """

    def __init__(self, num_classes: int):
        super(ModelNew, self).__init__()

        # Convolutional layer parameters (LeNet-5)
        self.conv1_weight = nn.Parameter(torch.empty(6, 1, 5, 5))
        self.conv1_bias = nn.Parameter(torch.empty(6))

        self.conv2_weight = nn.Parameter(torch.empty(16, 6, 5, 5))
        self.conv2_bias = nn.Parameter(torch.empty(16))

        # Fully connected layers
        self.fc1_weight = nn.Parameter(torch.empty(120, 16 * 5 * 5))
        self.fc1_bias = nn.Parameter(torch.empty(120))

        self.fc2_weight = nn.Parameter(torch.empty(84, 120))
        self.fc2_bias = nn.Parameter(torch.empty(84))

        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 84))
        self.fc3_bias = nn.Parameter(torch.empty(num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize like nn.Conv2d / nn.Linear (Kaiming uniform + bias uniform)

        # Conv1
        nn.init.kaiming_uniform_(self.conv1_weight, a=math.sqrt(5))
        fan_in = self.conv1_weight.shape[1] * self.conv1_weight.shape[2] * self.conv1_weight.shape[3]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1_bias, -bound, bound)

        # Conv2
        nn.init.kaiming_uniform_(self.conv2_weight, a=math.sqrt(5))
        fan_in = self.conv2_weight.shape[1] * self.conv2_weight.shape[2] * self.conv2_weight.shape[3]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv2_bias, -bound, bound)

        # FC1
        nn.init.kaiming_uniform_(self.fc1_weight, a=math.sqrt(5))
        fan_in = self.fc1_weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc1_bias, -bound, bound)

        # FC2
        nn.init.kaiming_uniform_(self.fc2_weight, a=math.sqrt(5))
        fan_in = self.fc2_weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc2_bias, -bound, bound)

        # FC3
        nn.init.kaiming_uniform_(self.fc3_weight, a=math.sqrt(5))
        fan_in = self.fc3_weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc3_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1, 32, 32)
        returns: (batch_size, num_classes)
        """
        # Conv1 + ReLU (Triton), then MaxPool
        x = conv2d_relu_triton(x, self.conv1_weight, self.conv1_bias)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv2 + ReLU (Triton), then MaxPool
        x = conv2d_relu_triton(x, self.conv2_weight, self.conv2_bias)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.shape[0], -1)  # (B, 16*5*5)

        # FC1 + ReLU (Triton)
        x = linear_bias_relu_triton(x, self.fc1_weight, self.fc1_bias)

        # FC2 + ReLU (Triton)
        x = linear_bias_relu_triton(x, self.fc2_weight, self.fc2_bias)

        # FC3 (Triton, no activation)
        x = linear_bias_triton(x, self.fc3_weight, self.fc3_bias)

        return x
