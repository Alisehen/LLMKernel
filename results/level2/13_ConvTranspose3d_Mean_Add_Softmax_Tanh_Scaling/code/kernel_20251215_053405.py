# Optimized Triton code for ConvTranspose3d + fused post-ops

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Mean over depth (D)  ->  mean[b, c, h, w] in fp32
# Memory-bound; keep simple and bandwidth-saturating.
# ---------------------------------------------------------------------------

@triton.jit
def mean_over_depth_kernel(
    x_ptr,         # [B, C, D, H, W]
    mean_ptr,      # [B, C, H, W] (fp32)
    B, C, D, H, W,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_mb, stride_mc, stride_mh, stride_mw,
    BLOCK_W: tl.constexpr,
):
    """
    Each program computes mean over D for a tile of W for fixed (b, c, h).

    Grid:
      pid0 = b * (C*H) + c * H + h   (0 .. B*C*H - 1)
      pid1 = w-tile index            (0 .. ceil_div(W, BLOCK_W) - 1)
    """
    pid_bch = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Decode (b, c, h)
    bc_h = C * H
    b = pid_bch // bc_h
    tmp = pid_bch % bc_h
    c = tmp // H
    h = tmp % H

    # Guards
    if b >= B:
        return
    if c >= C:
        return
    if h >= H:
        return

    # Tile of W this program handles
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    # Base pointers for this (b, c, h)
    base_x = (
        x_ptr
        + b * stride_xb
        + c * stride_xc
        + h * stride_xh
    )
    base_mean = (
        mean_ptr
        + b * stride_mb
        + c * stride_mc
        + h * stride_mh
    )

    # Accumulate over D in fp32
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for d in range(0, D):
        x_ptrs = base_x + d * stride_xd + offs_w * stride_xw
        x_vals = tl.load(x_ptrs, mask=mask_w, other=0.0)
        acc += x_vals.to(tl.float32)

    # Final mean
    mean_vals = acc / D

    # Store to mean tensor [B, C, H, W]
    mean_ptrs = base_mean + offs_w * stride_mw
    tl.store(mean_ptrs, mean_vals, mask=mask_w)


# ---------------------------------------------------------------------------
# Kernel 2: Bias + softmax over C + tanh + scaling
#           Input:  mean[b, c, h, w] (fp32)
#           Output: out[b, c, 0, h, w] (x.dtype)
#
# Optimizations:
#   - Fast path when C <= BLOCK_C:
#       * Load mean+bias once
#       * Compute max, exp, softmax, tanh, scale in a single pass
#       * Avoid re-loading and re-computing exp(v - max)
#   - Tiled path when C > BLOCK_C:
#       * 2-pass numerically stable log-sum-exp streaming reduction
#       * Reduces from 3 passes (original code) to 2 passes
#   - Autotune over num_warps / num_stages per current optimization stage:
#       * Baseline: num_warps=4, num_stages=2
#       * Aggressive: num_warps=8, num_stages=3 (for low register pressure)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Baseline / conservative config
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        # More aggressive config for compute-bound fusion
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=3),
    ],
    key=['C'],
)
@triton.jit
def fused_bias_softmax_tanh_scale_kernel(
    mean_ptr,      # [B, C, H, W] (fp32)
    bias_ptr,      # [C]          (same dtype as out)
    out_ptr,       # [B, C, 1, H, W]
    B, C, H, W,
    stride_mb, stride_mc, stride_mh, stride_mw,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    """
    For each (b, h, w), compute:
      v_c = mean[b, c, h, w] + bias[c]
      softmax over c: s_c = exp(v_c - max_v) / sum_c exp(v_c - max_v)
      y_c = scaling_factor * tanh(s_c)

    Grid:
      pid = b * (H*W) + h * W + w  (0 .. B*H*W - 1)
      Each program handles all C (possibly in BLOCK_C tiles) for one (b, h, w).
    """
    pid = tl.program_id(0)
    bhw = H * W
    b = pid // bhw
    hw = pid % bhw
    h = hw // W
    w = hw % W

    # Guards
    if b >= B:
        return
    if h >= H:
        return
    if w >= W:
        return

    offs_c = tl.arange(0, BLOCK_C)
    neg_inf = -float('inf')

    # Base pointers for this (b, h, w)
    base_mean = (
        mean_ptr
        + b * stride_mb
        + h * stride_mh
        + w * stride_mw
    )
    base_out = (
        out_ptr
        + b * stride_ob
        + h * stride_oh
        + w * stride_ow
    )

    scale = tl.full((), scaling_factor, dtype=tl.float32)

    # ------------------------------------------------------------------ #
    # Fast path: C fits in a single BLOCK_C tile                         #
    #   - Single pass over C                                             #
    #   - Only one exp for softmax, one exp for tanh per element         #
    # ------------------------------------------------------------------ #
    if C <= BLOCK_C:
        c_idxs = offs_c
        mask_c = c_idxs < C

        mean_ptrs = base_mean + c_idxs * stride_mc
        mean_vals = tl.load(mean_ptrs, mask=mask_c, other=0.0)  # fp32

        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        bias_vals = bias_vals.to(tl.float32)

        v = mean_vals + bias_vals
        v = tl.where(mask_c, v, neg_inf)

        # Softmax in one pass
        max_v = tl.max(v, axis=0)
        v_shifted = v - max_v
        numer = tl.exp(v_shifted)
        denom = tl.sum(numer, axis=0)
        softmax_vals = numer / denom

        # tanh via stable exponential form: tanh(x) = (1 - e^{-2x}) / (1 + e^{-2x})
        e_neg_2x = tl.exp(-2.0 * softmax_vals)
        tanh_vals = (1.0 - e_neg_2x) / (1.0 + e_neg_2x)

        out_vals = tanh_vals * scale

        out_ptrs = base_out + c_idxs * stride_oc
        tl.store(out_ptrs, out_vals, mask=mask_c)
        return

    # ------------------------------------------------------------------ #
    # Tiled path: C > BLOCK_C                                            #
    #   - 2-pass numerically stable log-sum-exp across tiles             #
    #   - Pass 1: streaming reduction for global max and denominator     #
    #   - Pass 2: compute softmax, tanh, scale, and store                #
    # ------------------------------------------------------------------ #

    # Pass 1: streaming log-sum-exp over C (compute global max and sum_exp)
    m = tl.full((), neg_inf, dtype=tl.float32)
    s = tl.zeros((), dtype=tl.float32)

    c0 = 0
    while c0 < C:
        c_idxs = c0 + offs_c
        mask_c = c_idxs < C

        mean_ptrs = base_mean + c_idxs * stride_mc
        mean_vals = tl.load(mean_ptrs, mask=mask_c, other=0.0)  # fp32

        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        bias_vals = bias_vals.to(tl.float32)

        v = mean_vals + bias_vals
        v = tl.where(mask_c, v, neg_inf)

        local_max = tl.max(v, axis=0)
        v_shifted_local = v - local_max
        exp_vals = tl.exp(v_shifted_local)
        local_sum = tl.sum(exp_vals, axis=0)

        # Merge (local_max, local_sum) into global (m, s)
        new_m = tl.maximum(m, local_max)
        # s * exp(m - new_m) + local_sum * exp(local_max - new_m)
        s = s * tl.exp(m - new_m) + local_sum * tl.exp(local_max - new_m)
        m = new_m

        c0 += BLOCK_C

    inv_denom = 1.0 / s

    # Pass 2: compute softmax, tanh, scaling, and store
    c0 = 0
    while c0 < C:
        c_idxs = c0 + offs_c
        mask_c = c_idxs < C

        mean_ptrs = base_mean + c_idxs * stride_mc
        mean_vals = tl.load(mean_ptrs, mask=mask_c, other=0.0)

        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        bias_vals = bias_vals.to(tl.float32)

        v = mean_vals + bias_vals
        v = tl.where(mask_c, v, neg_inf)

        # Softmax using precomputed global max m and denominator
        v_shifted = v - m
        numer = tl.exp(v_shifted)
        softmax_vals = numer * inv_denom

        # tanh via stable exponential form
        e_neg_2x = tl.exp(-2.0 * softmax_vals)
        tanh_vals = (1.0 - e_neg_2x) / (1.0 + e_neg_2x)

        out_vals = tanh_vals * scale

        out_ptrs = base_out + c_idxs * stride_oc
        tl.store(out_ptrs, out_vals, mask=mask_c)

        c0 += BLOCK_C


# ---------------------------------------------------------------------------
# Wrapper for launching kernels
# ---------------------------------------------------------------------------

def fused_post_convtranspose_3d(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    x:    [B, C, D, H, W] - output of ConvTranspose3d
    bias: [1, C, 1, 1, 1] - broadcastable bias
    return: [B, C, 1, H, W] after mean(D) + bias + softmax(C) + tanh + scaling
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels"
    B, C, D, H, W = x.shape

    # Per-channel bias as [C]
    bias_flat = bias.view(-1)  # [C]

    # Intermediate mean over depth in fp32 for better numerical stability
    mean = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

    # Final output (same dtype as x)
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)

    # ---------------------- Launch mean_over_depth_kernel -------------------
    BLOCK_W = 64  # good default for contiguous W dimension

    grid_mean = (B * C * H, triton.cdiv(W, BLOCK_W))
    mean_over_depth_kernel[grid_mean](
        x,
        mean,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        mean.stride(0), mean.stride(1), mean.stride(2), mean.stride(3),
        BLOCK_W=BLOCK_W,
    )

    # ---------------- Launch fused_bias_softmax_tanh_scale_kernel ----------
    grid_softmax = (B * H * W,)

    fused_bias_softmax_tanh_scale_kernel[grid_softmax](
        mean,
        bias_flat,
        out,
        B, C, H, W,
        mean.stride(0), mean.stride(1), mean.stride(2), mean.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        float(scaling_factor),
    )

    return out


# ---------------------------------------------------------------------------
# PyTorch module integrating ConvTranspose3d with optimized Triton kernels
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    PyTorch ConvTranspose3d + Triton-optimized post-ops:

      1. ConvTranspose3d  (PyTorch native)
      2. Mean over depth (D)       [Triton kernel]
      3. Bias add                  \
      4. Softmax over channels (C)  > fused in Triton kernel
      5. Tanh                      /
      6. Scaling                   [Triton kernel]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)  # [B, C, D, H, W]
        x = fused_post_convtranspose_3d(x, self.bias, self.scaling_factor)
        return x
