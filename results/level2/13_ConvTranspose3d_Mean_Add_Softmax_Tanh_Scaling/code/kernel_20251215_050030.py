import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_mean_bias_softmax_tanh_scale_3d_kernel(
    x_ptr,          # [B, C, D, H, W]
    bias_ptr,       # [C]
    out_ptr,        # [B, C, 1, H, W]
    B, C, D, H, W,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    """
    Fused operations on ConvTranspose3d output:
      1. Mean over depth (D)
      2. Bias add (per channel)
      3. Softmax over channels (C)
      4. Tanh activation
      5. Scaling

    Input:  x [B, C, D, H, W]
    Output: out [B, C, 1, H, W]
    Softmax is computed over C for each (b, h, w).
    """
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Guard against out-of-bounds program ids
    if pid_w >= W:
        return

    # Decompose pid_bh into (b, h)
    b = pid_bh // H
    h = pid_bh % H

    if b >= B or h >= H:
        return

    offs_c = tl.arange(0, BLOCK_C)
    neg_inf = -float("inf")

    # -------------------------------------------------------------------------
    # Pass 1: compute global max over channels for softmax (per (b, h, w))
    # -------------------------------------------------------------------------
    max_val = neg_inf

    for c0 in range(0, C, BLOCK_C):
        c_idxs = c0 + offs_c
        mask_c = c_idxs < C

        # Mean over depth
        mean_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for d in range(0, D):
            x_ptrs = (
                x_ptr
                + b * stride_xb
                + c_idxs * stride_xc
                + d * stride_xd
                + h * stride_xh
                + pid_w * stride_xw
            )
            x_vals = tl.load(x_ptrs, mask=mask_c, other=0.0)
            mean_vals += x_vals
        mean_vals = mean_vals / D

        # Add bias
        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        vals = mean_vals + bias_vals

        # Masked max
        vals = tl.where(mask_c, vals, neg_inf)
        local_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, local_max)

    # -------------------------------------------------------------------------
    # Pass 2: compute denominator (sum of exp) for softmax
    # -------------------------------------------------------------------------
    sum_exp = 0.0

    for c0 in range(0, C, BLOCK_C):
        c_idxs = c0 + offs_c
        mask_c = c_idxs < C

        mean_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for d in range(0, D):
            x_ptrs = (
                x_ptr
                + b * stride_xb
                + c_idxs * stride_xc
                + d * stride_xd
                + h * stride_xh
                + pid_w * stride_xw
            )
            x_vals = tl.load(x_ptrs, mask=mask_c, other=0.0)
            mean_vals += x_vals
        mean_vals = mean_vals / D

        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        vals = mean_vals + bias_vals

        exp_vals = tl.exp(vals - max_val)
        exp_vals = tl.where(mask_c, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals, axis=0)

    inv_denom = 1.0 / sum_exp

    # -------------------------------------------------------------------------
    # Pass 3: compute final softmax, then tanh, then scaling, and store
    # -------------------------------------------------------------------------
    for c0 in range(0, C, BLOCK_C):
        c_idxs = c0 + offs_c
        mask_c = c_idxs < C

        mean_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for d in range(0, D):
            x_ptrs = (
                x_ptr
                + b * stride_xb
                + c_idxs * stride_xc
                + d * stride_xd
                + h * stride_xh
                + pid_w * stride_xw
            )
            x_vals = tl.load(x_ptrs, mask=mask_c, other=0.0)
            mean_vals += x_vals
        mean_vals = mean_vals / D

        bias_vals = tl.load(bias_ptr + c_idxs, mask=mask_c, other=0.0)
        vals = mean_vals + bias_vals

        # Softmax
        exp_vals = tl.exp(vals - max_val)
        softmax_vals = exp_vals * inv_denom

        # Tanh via exponential definition: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
        e2x = tl.exp(2.0 * softmax_vals)
        tanh_vals = (e2x - 1.0) / (e2x + 1.0)

        # Scaling
        out_vals = tanh_vals * scaling_factor

        # Store at depth index 0 (keepdim on depth)
        out_ptrs = (
            out_ptr
            + b * stride_ob
            + c_idxs * stride_oc
            + h * stride_oh
            + pid_w * stride_ow
        )
        tl.store(out_ptrs, out_vals, mask=mask_c)


def fused_post_convtranspose_3d(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    x:     [B, C, D, H, W] - output of ConvTranspose3d
    bias:  [1, C, 1, 1, 1] - broadcastable bias
    return: [B, C, 1, H, W] after mean(D) + bias + softmax(C) + tanh + scaling
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernel"
    B, C, D, H, W = x.shape

    # Flatten bias to [C]
    bias_flat = bias.view(-1)  # [C]

    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)

    BLOCK_C = 64  # power-of-2 block size for channels

    grid = (B * H, W)
    fused_mean_bias_softmax_tanh_scale_3d_kernel[grid](
        x,
        bias_flat,
        out,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        scaling_factor,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    PyTorch ConvTranspose3d + Triton-fused post-ops:

      1. ConvTranspose3d  (PyTorch native)
      2. Mean over depth (D)
      3. Bias add
      4. Softmax over channels
      5. Tanh
      6. Scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native (indexing is complex)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        # Bias broadcastable over [B, C, D, H, W]; we only need per-channel bias
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # ConvTranspose3d (PyTorch)
        x = self.conv_transpose(x)  # [B, C, D, H, W]
        # Fused Triton kernel for remaining ops
        x = fused_post_convtranspose_3d(x, self.bias, self.scaling_factor)
        return x
