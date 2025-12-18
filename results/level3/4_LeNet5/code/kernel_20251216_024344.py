# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# High-performance 2D conv (5x5, stride=1, no padding) + bias + ReLU
# - Fully fused: only 1 global store (final output)
# - All intermediates kept in registers
# - Computation-centric (compute-bound on RTX 4090), unrolled over Cin/Kh/Kw
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # For larger batches / feature maps
        triton.Config(
            {'BLOCK_CO': 16, 'BLOCK_N': 128},
            num_warps=4,
            num_stages=2,
        ),
        # Medium tile – good trade-off between occupancy & register pressure
        triton.Config(
            {'BLOCK_CO': 16, 'BLOCK_N': 64},
            num_warps=4,
            num_stages=2,
        ),
        # Small tile – safest fallback
        triton.Config(
            {'BLOCK_CO': 8, 'BLOCK_N': 64},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['B', 'H_out', 'W_out', 'Cout'],
)
@triton.jit
def conv2d_relu_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, H, W, H_out, W_out,
    stride_h, stride_w,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    Cin: tl.constexpr, Cout: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_co = tl.program_id(0)  # tile over output channels
    pid_n = tl.program_id(1)   # tile over flattened (B * H_out * W_out)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    N_total = B * H_out * W_out
    n_per_b = H_out * W_out

    mask_co = offs_co < Cout
    mask_n = offs_n < N_total
    mask = mask_co[:, None] & mask_n[None, :]

    # Decode flattened n index -> (b, ho, wo)
    b = offs_n // n_per_b
    hw = offs_n % n_per_b
    ho = hw // W_out
    wo = hw % W_out

    # Precompute scaled spatial offsets & batch offset
    ho_scaled = ho * stride_h
    wo_scaled = wo * stride_w
    xb = b * stride_xb

    # Accumulator: [BLOCK_CO, BLOCK_N]
    acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)

    # Base per-output-channel pointer (vector over CO)
    w_co_base = w_ptr + offs_co * stride_wco

    # Give compiler alignment hints for better memory scheduling
    tl.multiple_of(offs_co, BLOCK_CO)
    tl.multiple_of(offs_n, BLOCK_N)

    # Fully unrolled loops over Cin, Kh, Kw (compile-time small)
    for ci in tl.static_range(0, Cin):
        w_ci_base = w_co_base + ci * stride_wci
        x_ci_base = x_ptr + ci * stride_xc

        for kh in tl.static_range(0, Kh):
            ih = ho_scaled + kh
            x_row = x_ci_base + xb + ih * stride_xh
            w_kh_base = w_ci_base + kh * stride_wkh

            for kw in tl.static_range(0, Kw):
                iw = wo_scaled + kw

                # Load input tile across N (broadcast over CO)
                x_ptrs = x_row + iw * stride_xw
                x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)

                # Load weight tile across CO (broadcast over N)
                w_ptrs = w_kh_base + kw * stride_wkw
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0)

                # Outer-product update (kept entirely in registers)
                acc += w_vals[:, None] * x_vals[None, :]

    # Fused bias add (broadcast over N)
    bias = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)
    acc += bias[:, None]

    # Fused ReLU
    acc = tl.maximum(acc, 0.0)

    # Store final output (only global store in this kernel)
    y_ptrs = y_ptr + (
        b[None, :] * stride_yb +
        offs_co[:, None] * stride_yc +
        ho[None, :] * stride_yh +
        wo[None, :] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask)


# ---------------------------------------------------------------------------
# High-performance matmul (x @ w^T) + bias (+ optional ReLU) for Linear layers
# - Fully fused: loads x, w, bias; single store of final output
# - Uses tl.dot (Tensor Cores via TF32 where possible)
# - Autotuned over BLOCK_M / BLOCK_N / BLOCK_K; grid depends on META
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # Larger tiles for bigger matrices
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Medium tiles – good for typical LeNet FC shapes
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Small tiles – increase grid size for very small matrices
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    USE_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment hints for better memory scheduling
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Initialize accumulator in FP32 for numeric robustness
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointers for this tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    k = 0
    # Main K-loop – mask on K only (M,N masks are constant per tile)
    while k < K:
        k_remaining = K - k

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Use Tensor Cores via TF32 when possible
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add (broadcast over M) – bias loaded once per tile
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional fused ReLU
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    # Final output store (single global store)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper: Conv2d + Bias + ReLU (5x5, stride=1, no padding, NCHW)
# - Uses META-dependent grid so autotune configs remain correct
# ---------------------------------------------------------------------------
def conv2d_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    NCHW conv2d with kernel=5x5, stride=1, no padding, fused with bias and ReLU.
    Autotuned BLOCK_CO / BLOCK_N for RTX 4090 (Ada, SM 8.9).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    weight = weight.contiguous()

    B, Cin, H, W = x.shape
    Cout, Cin_w, Kh, Kw = weight.shape
    assert Cin == Cin_w
    assert Kh == Kw == 5
    stride_h = 1
    stride_w = 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1

    y = torch.empty((B, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(Cout, META['BLOCK_CO']),
            triton.cdiv(B * H_out * W_out, META['BLOCK_N']),
        )

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
    )

    return y


# ---------------------------------------------------------------------------
# Wrapper: Linear + Bias + ReLU  (x @ weight.T + bias)
# ---------------------------------------------------------------------------
def linear_bias_relu_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused x @ weight.T + bias with ReLU.
    x:      (M, K)
    weight: (N, K)  (PyTorch layout)  -> internally use weight.T of shape (K, N)
    Autotuned BLOCK_M / BLOCK_N / BLOCK_K for RTX 4090.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    w_t = weight.t().contiguous()  # (K, N)

    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

    linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        USE_RELU=True,
    )

    return y


# ---------------------------------------------------------------------------
# Wrapper: Linear + Bias (no activation)  (x @ weight.T + bias)
# ---------------------------------------------------------------------------
def linear_bias_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused x @ weight.T + bias (no activation).
    x:      (M, K)
    weight: (N, K)  (PyTorch layout) -> internally use weight.T (K, N)
    Autotuned BLOCK_M / BLOCK_N / BLOCK_K for RTX 4090.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x = x.contiguous()
    w_t = weight.t().contiguous()  # (K, N)

    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

    linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        USE_RELU=False,
    )

    return y


# ---------------------------------------------------------------------------
# Model definition using the optimized Triton kernels
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
        fan_in = (
            self.conv1_weight.shape[1]
            * self.conv1_weight.shape[2]
            * self.conv1_weight.shape[3]
        )
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1_bias, -bound, bound)

        # Conv2
        nn.init.kaiming_uniform_(self.conv2_weight, a=math.sqrt(5))
        fan_in = (
            self.conv2_weight.shape[1]
            * self.conv2_weight.shape[2]
            * self.conv2_weight.shape[3]
        )
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
        import torch.nn.functional as F

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
