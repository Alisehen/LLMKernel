import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# =========================
# Triton Conv2d (NCHW, stride=1, padding=0) + bias
# =========================

@triton.jit
def conv2d_nchw_kernel(
    x_ptr,        # *float32, [N, C_in, H, W]
    w_ptr,        # *float32, [C_out, K_total]  (C_out, C_in * K_h * K_w)
    b_ptr,        # *float32, [C_out]
    y_ptr,        # *float32, [N, C_out, H_out, W_out]
    N, C_in, H, W,
    C_out, K_h, K_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wk,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile in output positions (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # tile in output channels
    BLOCK_K: tl.constexpr,  # tile in K = C_in * K_h * K_w
):
    pid_m = tl.program_id(0)  # over output positions (N * H_out * W_out)
    pid_n = tl.program_id(1)  # over output channels

    M = N * H_out * W_out
    K_total = C_in * K_h * K_w
    hw_out = H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decompose linear output index -> (n, h_out, w_out)
    n_idx = offs_m // hw_out
    rem_hw = offs_m % hw_out
    ho = rem_hw // W_out
    wo = rem_hw % W_out

    n_idx = n_idx[:, None]
    ho = ho[:, None]
    wo = wo[:, None]
    offs_n_b = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        ci = offs_k // (K_h * K_w)
        rem_kk = offs_k % (K_h * K_w)
        ky = rem_kk // K_w
        kx = rem_kk % K_w

        ci_b = ci[None, :]
        ky_b = ky[None, :]
        kx_b = kx[None, :]

        hi = ho + ky_b
        wi = wo + kx_b

        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + ci_b * stride_xc
            + hi * stride_xh
            + wi * stride_xw
        )
        x_block = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        w_ptrs = w_ptr + offs_n_b * stride_wn + offs_k[:, None] * stride_wk
        w_block = tl.load(
            w_ptrs,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )

        acc += tl.dot(x_block, w_block, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    y_ptrs = (
        y_ptr
        + n_idx * stride_yn
        + offs_n_b * stride_yc
        + ho * stride_yh
        + wo * stride_yw
    )
    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def conv2d_triton(x, weight, bias):
    """
    x:      (N, C_in, H, W)
    weight: (C_out, C_in, K_h, K_w)
    bias:   (C_out,)
    """
    assert x.ndim == 4
    assert weight.ndim == 4
    N, C_in, H, W = x.shape
    C_out, C_w, K_h, K_w = weight.shape
    assert C_w == C_in

    # Stride=1, padding=0, dilation=1
    H_out = H - K_h + 1
    W_out = W - K_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten weight to [C_out, K_total]
    w_flat = weight.view(C_out, -1).contiguous()

    M = N * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, w_flat, bias, y,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_flat.stride(0), w_flat.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return y


# =========================
# Triton Mish + per-channel stats (unfused, kept for reference)
# =========================

@triton.jit
def mish_and_reduce_kernel(
    x_ptr, y_ptr,
    sum_ptr, sumsq_ptr,   # [C]
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,   # S = N * H * W (flattened)
):
    pid_c = tl.program_id(0)  # channel tiles
    pid_s = tl.program_id(1)  # spatial+batch tiles

    S = N * H * W
    hw = H * W

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_c = offs_c < C
    mask_s = offs_s < S

    n_idx = offs_s // hw
    rem_hw = offs_s % hw
    h_idx = rem_hw // W
    w_idx = rem_hw % W

    n_idx = n_idx[:, None]
    h_idx = h_idx[:, None]
    w_idx = w_idx[:, None]
    offs_c_b = offs_c[None, :]

    x_ptrs = (
        x_ptr
        + n_idx * stride_xn
        + offs_c_b * stride_xc
        + h_idx * stride_xh
        + w_idx * stride_xw
    )
    x_block = tl.load(
        x_ptrs,
        mask=mask_s[:, None] & mask_c[None, :],
        other=0.0,
    )

    # Mish(x) = x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    e = tl.exp(x_block)
    sp = tl.log(1.0 + e)
    t = tl.exp(2.0 * sp)
    tanh_sp = (t - 1.0) / (t + 1.0)
    mish = x_block * tanh_sp

    tl.store(
        y_ptr
        + n_idx * stride_yn
        + offs_c_b * stride_yc
        + h_idx * stride_yh
        + w_idx * stride_yw,
        mish,
        mask=mask_s[:, None] & mask_c[None, :],
    )

    # Reduce over S dimension -> per-channel partial sums
    partial_sum = tl.sum(mish, axis=0)
    partial_sumsq = tl.sum(mish * mish, axis=0)

    tl.atomic_add(sum_ptr + offs_c, partial_sum, mask=mask_c)
    tl.atomic_add(sumsq_ptr + offs_c, partial_sumsq, mask=mask_c)


def mish_with_stats_triton(x):
    """
    x: (N, C, H, W)
    Returns:
      y      : Mish(x)
      sum    : per-channel sum of y over N,H,W
      sum_sq : per-channel sum of y^2 over N,H,W
    """
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    sum_ = torch.zeros(C, device=x.device, dtype=torch.float32)
    sumsq_ = torch.zeros_like(sum_)

    S = N * H * W
    grid = lambda META: (
        triton.cdiv(C, META["BLOCK_C"]),
        triton.cdiv(S, META["BLOCK_S"]),
    )

    mish_and_reduce_kernel[grid](
        x, y,
        sum_, sumsq_,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_C=64,
        BLOCK_S=256,
        num_warps=4,
        num_stages=2,
    )
    return y, sum_, sumsq_


# =========================
# FUSED Triton Conv2d + Mish + per-channel stats
# =========================

@triton.jit
def conv2d_mish_reduce_nchw_kernel(
    x_ptr,        # *float32, [N, C_in, H, W]
    w_ptr,        # *float32, [C_out, K_total]
    b_ptr,        # *float32, [C_out]
    y_ptr,        # *float32, [N, C_out, H_out, W_out]  (Mish output)
    sum_ptr,      # *float32, [C_out]  per-channel sum over N,H_out,W_out
    sumsq_ptr,    # *float32, [C_out]  per-channel sum of squares
    N, C_in, H, W,
    C_out, K_h, K_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wk,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile in output positions (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # tile in output channels
    BLOCK_K: tl.constexpr,  # tile in K = C_in * K_h * K_w
):
    pid_m = tl.program_id(0)  # over output positions (N * H_out * W_out)
    pid_n = tl.program_id(1)  # over output channels

    M = N * H_out * W_out
    K_total = C_in * K_h * K_w
    hw_out = H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out
    mask_full = mask_m[:, None] & mask_n[None, :]

    # Decompose linear output index -> (n, h_out, w_out)
    n_idx = offs_m // hw_out
    rem_hw = offs_m % hw_out
    ho = rem_hw // W_out
    wo = rem_hw % W_out

    n_idx = n_idx[:, None]
    ho = ho[:, None]
    wo = wo[:, None]
    offs_n_b = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        ci = offs_k // (K_h * K_w)
        rem_kk = offs_k % (K_h * K_w)
        ky = rem_kk // K_w
        kx = rem_kk % K_w

        ci_b = ci[None, :]
        ky_b = ky[None, :]
        kx_b = kx[None, :]

        hi = ho + ky_b
        wi = wo + kx_b

        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + ci_b * stride_xc
            + hi * stride_xh
            + wi * stride_xw
        )
        x_block = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        w_ptrs = w_ptr + offs_n_b * stride_wn + offs_k[:, None] * stride_wk
        w_block = tl.load(
            w_ptrs,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        )

        acc += tl.dot(x_block, w_block, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Mish activation: Mish(x) = x * tanh(softplus(x)),
    # softplus(x) = log(1 + exp(x))
    e = tl.exp(acc)
    sp = tl.log(1.0 + e)
    t = tl.exp(2.0 * sp)
    tanh_sp = (t - 1.0) / (t + 1.0)
    mish = acc * tanh_sp

    # Mask out invalid lanes so they don't affect reduction
    mish = tl.where(mask_full, mish, 0.0)

    # Store Mish output
    y_ptrs = (
        y_ptr
        + n_idx * stride_yn
        + offs_n_b * stride_yc
        + ho * stride_yh
        + wo * stride_yw
    )
    tl.store(
        y_ptrs,
        mish,
        mask=mask_full,
    )

    # Per-channel partial sums over this BLOCK_M tile
    partial_sum = tl.sum(mish, axis=0)          # shape: [BLOCK_N]
    partial_sumsq = tl.sum(mish * mish, axis=0)

    # Atomically accumulate into global buffers
    tl.atomic_add(sum_ptr + offs_n, partial_sum, mask=mask_n)
    tl.atomic_add(sumsq_ptr + offs_n, partial_sumsq, mask=mask_n)


def conv2d_mish_with_stats_triton(x, weight, bias):
    """
    Fused Conv2d (stride=1, padding=0) + Mish + per-channel stats.

    x:      (N, C_in, H, W)
    weight: (C_out, C_in, K_h, K_w)
    bias:   (C_out,)

    Returns:
      y      : Mish(conv2d(x, weight, bias))  (N, C_out, H_out, W_out)
      sum    : per-channel sum of y over N,H_out,W_out  (C_out,)
      sum_sq : per-channel sum of y^2 over N,H_out,W_out  (C_out,)
    """
    assert x.ndim == 4
    assert weight.ndim == 4
    N, C_in, H, W = x.shape
    C_out, C_w, K_h, K_w = weight.shape
    assert C_w == C_in

    # Stride=1, padding=0, dilation=1
    H_out = H - K_h + 1
    W_out = W - K_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    sum_ = torch.zeros(C_out, device=x.device, dtype=torch.float32)
    sumsq_ = torch.zeros_like(sum_)

    # Flatten weight to [C_out, K_total]
    w_flat = weight.view(C_out, -1).contiguous()

    M = N * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_mish_reduce_nchw_kernel[grid](
        x, w_flat, bias, y, sum_, sumsq_,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_flat.stride(0), w_flat.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return y, sum_, sumsq_


# =========================
# Triton BatchNorm (per-channel NCHW)
# =========================

@triton.jit
def batch_norm_kernel(
    x_ptr, y_ptr,
    mean_ptr, var_ptr,
    weight_ptr, bias_ptr,
    N, C, H, W,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)

    S = N * H * W
    hw = H * W

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_c = offs_c < C
    mask_s = offs_s < S

    n_idx = offs_s // hw
    rem_hw = offs_s % hw
    h_idx = rem_hw // W
    w_idx = rem_hw % W

    n_idx = n_idx[:, None]
    h_idx = h_idx[:, None]
    w_idx = w_idx[:, None]
    offs_c_b = offs_c[None, :]

    x_ptrs = (
        x_ptr
        + n_idx * stride_xn
        + offs_c_b * stride_xc
        + h_idx * stride_xh
        + w_idx * stride_xw
    )
    x_block = tl.load(
        x_ptrs,
        mask=mask_s[:, None] & mask_c[None, :],
        other=0.0,
    )

    mean = tl.load(mean_ptr + offs_c, mask=mask_c, other=0.0)
    var = tl.load(var_ptr + offs_c, mask=mask_c, other=1.0)
    gamma = tl.load(weight_ptr + offs_c, mask=mask_c, other=1.0)
    beta = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    mean = mean[None, :]
    var = var[None, :]
    gamma = gamma[None, :]
    beta = beta[None, :]

    inv_std = 1.0 / tl.sqrt(var + eps)
    y_block = (x_block - mean) * inv_std
    y_block = y_block * gamma + beta

    y_ptrs = (
        y_ptr
        + n_idx * stride_yn
        + offs_c_b * stride_yc
        + h_idx * stride_yh
        + w_idx * stride_yw
    )
    tl.store(
        y_ptrs,
        y_block,
        mask=mask_s[:, None] & mask_c[None, :],
    )


def batch_norm_triton(x, mean, var, weight, bias, eps):
    """
    x:      (N, C, H, W)
    mean:   (C,)
    var:    (C,)
    weight: (C,)
    bias:   (C,)
    """
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    S = N * H * W
    grid = lambda META: (
        triton.cdiv(C, META["BLOCK_C"]),
        triton.cdiv(S, META["BLOCK_S"]),
    )

    batch_norm_kernel[grid](
        x, y,
        mean, var,
        weight, bias,
        N, C, H, W,
        eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_C=64,
        BLOCK_S=256,
        num_warps=4,
        num_stages=2,
    )
    return y


# =========================
# ModelNew
# =========================

class ModelNew(nn.Module):
    """
    Conv2d (stride=1, padding=0) -> Mish -> BatchNorm2d,
    with Conv2d and Mish+stats fused in a single high-performance Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1]
            k = kernel_size[0]
        else:
            k = int(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k

        # Conv2d parameters (no padding, stride=1, bias=True)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # BatchNorm2d parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = True  # mimic nn.BatchNorm2d default

    def forward(self, x):
        # Ensure contiguous tensors on the correct device
        x = x.to(self.weight.device)
        x = x.contiguous()

        # 1) Fused Convolution + Mish activation + per-channel stats for BatchNorm
        mish_out, sum_, sumsq_ = conv2d_mish_with_stats_triton(x, self.weight, self.bias)

        N, C, H, W = mish_out.shape
        m = N * H * W

        # Compute batch mean and (biased) variance per channel
        batch_mean = sum_ / m
        batch_var = sumsq_ / m - batch_mean * batch_mean

        # Update running stats (training mode with tracking)
        if self.track_running_stats and self.training:
            if self.momentum is None:
                # Fallback: simple running average with factor 1/N
                momentum = 1.0 / m
            else:
                momentum = self.momentum

            with torch.no_grad():
                # Unbiased variance estimator for running_var
                if m > 1:
                    unbiased_var = batch_var * (m / (m - 1.0))
                else:
                    unbiased_var = batch_var

                self.running_mean.mul_(1.0 - momentum).add_(momentum * batch_mean)
                self.running_var.mul_(1.0 - momentum).add_(momentum * unbiased_var)

        # Choose stats for normalization
        if self.training or not self.track_running_stats:
            mean_use = batch_mean
            var_use = batch_var
        else:
            mean_use = self.running_mean
            var_use = self.running_var

        # 2) BatchNorm on Mish output
        out = batch_norm_triton(
            mish_out,
            mean_use,
            var_use,
            self.bn_weight,
            self.bn_bias,
            self.eps,
        )
        return out
