# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Optimized Triton kernels
# ------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_CI': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_CI': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_CI': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_CI': 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=['N', 'C_in', 'H', 'W', 'C_out'],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr,         # float32*  [N, C_in, H, W]
    w_ptr,         # float32*  [C_out, C_in, K, K]
    b_ptr,         # float32*  [C_out]
    y_ptr,         # float32*  [N, C_out, H_out, W_out]

    N, C_in, H, W, C_out,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,

    K: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    """
    NCHW convolution, stride=1, dilation=1, square kernel, symmetric padding.
    Fusion: conv + bias add.
    Grid:
      - program_id(0): tiles over M = N * H_out * W_out
      - program_id(1): tiles over N = C_out
    All fused ops (bias, store) share the same (offs_m, offs_n, out_mask).
    """
    # Output spatial dimensions (stride=1, dilation=1)
    H_out = H + 2 * PAD - K + 1
    W_out = W + 2 * PAD - K + 1
    HW_out = H_out * W_out
    S = N * HW_out  # total number of output positions (M dimension)

    pid_m = tl.program_id(0)  # over N*H_out*W_out
    pid_n = tl.program_id(1)  # over C_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < S
    n_mask = offs_n < C_out
    out_mask = m_mask[:, None] & n_mask[None, :]

    # Decode (n, h_out, w_out) from a flat index M
    n = offs_m // HW_out
    hw_rem = offs_m % HW_out
    h_out = hw_rem // W_out
    w_out = hw_rem % W_out

    # Hints for Triton to optimize address calculations
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Accumulator for output tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels in chunks of BLOCK_CI
    ci_range = tl.arange(0, BLOCK_CI)
    for ci0 in range(0, C_in, BLOCK_CI):
        ci = ci0 + ci_range
        ci_mask = ci < C_in

        # Base pointer for inputs for this (n, ci) tile (no kernel offsets yet)
        # Shape: [BLOCK_M, BLOCK_CI]
        x_base_nc = (
            x_ptr
            + n[:, None] * stride_x_n
            + ci[None, :] * stride_x_c
        )

        # Base pointer for weights for this (ci, oc) tile (no kernel offsets yet)
        # Shape: [BLOCK_CI, BLOCK_N]
        w_base_ci_oc = (
            w_ptr
            + ci[:, None] * stride_w_ic
            + offs_n[None, :] * stride_w_oc
        )

        # Mask for weights (only depends on ci and oc)
        mask_w_base = ci_mask[:, None] & n_mask[None, :]

        # Iterate over kernel window K x K
        for kh in range(0, K):
            # Map output position to input row with padding
            h_in = h_out + kh - PAD
            h_in_bounds = (h_in >= 0) & (h_in < H) & m_mask

            # Pre-apply row offset
            x_base_nch = x_base_nc + h_in[:, None] * stride_x_h

            for kw in range(0, K):
                # Map output position to input column with padding
                w_in = w_out + kw - PAD
                in_bounds = (
                    h_in_bounds
                    & (w_in >= 0)
                    & (w_in < W)
                )

                # Input pointers: [BLOCK_M, BLOCK_CI]
                x_ptrs = x_base_nch + w_in[:, None] * stride_x_w
                mask_x = in_bounds[:, None] & ci_mask[None, :]

                x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # Weight pointers: [BLOCK_CI, BLOCK_N]
                w_ptrs = (
                    w_base_ci_oc
                    + kh * stride_w_kh
                    + kw * stride_w_kw
                )
                w_vals = tl.load(w_ptrs, mask=mask_w_base, other=0.0)

                # Multiply-accumulate: (BLOCK_M, BLOCK_CI) x (BLOCK_CI, BLOCK_N)
                acc += tl.dot(x_vals, w_vals)

    # Fused bias add (same offsets and mask as store along C_out dimension)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Store result: [BLOCK_M, BLOCK_N]
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_c
        + h_out[:, None] * stride_y_h
        + w_out[:, None] * stride_y_w
    )
    tl.store(y_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_HW': 128, 'BLOCK_C': 64}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def maxpool2d_3x3_s1_p1_kernel(
    x_ptr,  # float32* [N, C, H, W]
    y_ptr,  # float32* [N, C, H, W]
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    3x3 max pooling, stride=1, padding=1, NCHW layout.
    Grid:
      - program_id(0): tiles over M = N * H * W
      - program_id(1): tiles over C
    Fused ops (max reduction + store) share the same output offsets & mask.
    """
    # 3x3, stride=1, padding=1 -> H_out = H, W_out = W
    H_out = H
    W_out = W
    HW_out = H_out * W_out
    S = N * HW_out

    pid_m = tl.program_id(0)  # over N*H*W
    pid_c = tl.program_id(1)  # over C

    offs_m = pid_m * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    m_mask = offs_m < S
    c_mask = offs_c < C
    out_mask = m_mask[:, None] & c_mask[None, :]

    # Decode (n, h_out, w_out) from flat index
    n = offs_m // HW_out
    hw_rem = offs_m % HW_out
    h_out = hw_rem // W_out
    w_out = hw_rem % W_out

    tl.multiple_of(offs_m, BLOCK_HW)
    tl.multiple_of(offs_c, BLOCK_C)

    # Initialize with -inf
    val_max = tl.full((BLOCK_HW, BLOCK_C), -float('inf'), dtype=tl.float32)

    # Base pointer for (n, c) (no spatial offsets yet): [BLOCK_HW, BLOCK_C]
    x_base_nc = (
        x_ptr
        + n[:, None] * stride_x_n
        + offs_c[None, :] * stride_x_c
    )

    # Pool over 3x3 window
    for kh in range(0, 3):
        for kw in range(0, 3):
            h_in = h_out + kh - 1
            w_in = w_out + kw - 1

            in_bounds = (
                (h_in >= 0)
                & (h_in < H)
                & (w_in >= 0)
                & (w_in < W)
                & m_mask
            )

            x_ptrs = (
                x_base_nc
                + h_in[:, None] * stride_x_h
                + w_in[:, None] * stride_x_w
            )
            mask_x = in_bounds[:, None] & c_mask[None, :]

            vals = tl.load(x_ptrs, mask=mask_x, other=-float('inf'))
            val_max = tl.maximum(val_max, vals)

    # Store result
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_y_n
        + offs_c[None, :] * stride_y_c
        + h_out[:, None] * stride_y_h
        + w_out[:, None] * stride_y_w
    )
    tl.store(y_ptrs, val_max, mask=out_mask)


# ------------------------------------------------------------
# Wrapper functions
# ------------------------------------------------------------

def triton_conv2d_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int):
    """
    NCHW convolution, stride=1, dilation=1, square kernel, symmetric padding.
    x:      [N, C_in, H, W]
    weight: [C_out, C_in, K, K]
    bias:   [C_out]
    """
    if not x.is_cuda:
        # Fallback for non-CUDA tensors
        return torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=padding, dilation=1)

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, K, K_w = weight.shape
    assert C_in_w == C_in and K == K_w, "Incompatible weight shape"

    H_out = H + 2 * padding - K + 1
    W_out = W + 2 * padding - K + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(N * H_out * W_out, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        K=K,
        PAD=padding,
    )

    return y


def triton_maxpool2d_3x3_s1_p1(x: torch.Tensor):
    """
    3x3 max pooling, stride=1, padding=1, NCHW layout.
    """
    if not x.is_cuda:
        return torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    def grid(meta):
        return (
            triton.cdiv(N * H * W, meta['BLOCK_HW']),
            triton.cdiv(C, meta['BLOCK_C']),
        )

    maxpool2d_3x3_s1_p1_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y


# ------------------------------------------------------------
# Model using optimized Triton kernels
# ------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, pool_proj):
        """
        Inception-style block with Triton-accelerated convolutions and pooling.
        Parameter structure matches the original Model so that state_dict
        is compatible.
        """
        super(ModelNew, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure NCHW contiguous for Triton
        x = x.contiguous()

        # Branch 1: 1x1 conv
        w1 = self.branch1x1.weight
        b1 = self.branch1x1.bias
        branch1x1 = triton_conv2d_nchw(x, w1, b1, padding=0)

        # Branch 2: 1x1 reduction -> 3x3 conv
        w3r = self.branch3x3[0].weight
        b3r = self.branch3x3[0].bias
        x3 = triton_conv2d_nchw(x, w3r, b3r, padding=0)

        w3 = self.branch3x3[1].weight
        b3 = self.branch3x3[1].bias
        branch3x3 = triton_conv2d_nchw(x3, w3, b3, padding=1)

        # Branch 3: 1x1 reduction -> 5x5 conv
        w5r = self.branch5x5[0].weight
        b5r = self.branch5x5[0].bias
        x5 = triton_conv2d_nchw(x, w5r, b5r, padding=0)

        w5 = self.branch5x5[1].weight
        b5 = self.branch5x5[1].bias
        branch5x5 = triton_conv2d_nchw(x5, w5, b5, padding=2)

        # Branch 4: 3x3 max pool -> 1x1 conv
        pooled = triton_maxpool2d_3x3_s1_p1(x)

        wp = self.branch_pool[1].weight
        bp = self.branch_pool[1].bias
        branch_pool = triton_conv2d_nchw(pooled, wp, bp, padding=0)

        # Concatenate along channel dimension
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)
