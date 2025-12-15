import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================
# Optimized Triton Kernels
# ============================

@triton.autotune(
    configs=[
        # UNROLL=1 for very small S
        triton.Config({'BLOCK_T': 128, 'UNROLL': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_T': 256, 'UNROLL': 1}, num_warps=4, num_stages=2),
        # UNROLL=2 with higher stages to increase ILP and hide memory latency
        triton.Config({'BLOCK_T': 256, 'UNROLL': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_T': 512, 'UNROLL': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_T': 1024, 'UNROLL': 2}, num_warps=8, num_stages=4),
    ],
    key=['S'],  # tune based on spatial size
)
@triton.jit
def subtract_spatial_mean_kernel_contiguous(
    x_ptr, out_ptr,
    NC, S,
    BLOCK_T: tl.constexpr,
    UNROLL: tl.constexpr,
):
    """
    Fast path for contiguous [N, C, D, H, W] flattened to [NC, S].

    For each (n,c) row: out[nc, s] = x[nc, s] - mean_s(x[nc, :])
    """
    pid = tl.program_id(axis=0)
    if pid >= NC:
        return

    base_x = x_ptr + pid * S
    base_out = out_ptr + pid * S

    offs = tl.arange(0, BLOCK_T)
    tl.multiple_of(offs, BLOCK_T)
    tl.max_contiguous(offs, BLOCK_T)

    # ---------------------------------------------------------------
    # Pass 1: compute mean over spatial dimension for this (n, c)
    # ---------------------------------------------------------------
    sum_val = tl.zeros((), dtype=tl.float32)

    step = BLOCK_T * UNROLL
    # Manually unrolled loop for higher ILP
    for start in range(0, S, step):
        # Inner unroll
        for u in range(UNROLL):
            idx = start + offs + u * BLOCK_T
            mask = idx < S
            x_vals = tl.load(
                base_x + idx,
                mask=mask,
                other=0.0,
                cache_modifier='.cg',  # streaming load: bypass L1, use L2
            )
            x_vals_f32 = x_vals.to(tl.float32)
            sum_val += tl.sum(x_vals_f32, axis=0)

    # Compute mean (S is integer; Triton promotes to float)
    mean = sum_val / S

    # Pre-cast mean once to avoid repeated conversions in the loop
    # scalar in same dtype as x
    # (We don't know x dtype here; assume same as out)
    # Just load one element to get dtype without extra parameter
    # NOTE: all rows share same dtype, so this is cheap
    x0 = tl.load(base_x + 0)
    mean_cast = tl.cast(mean, x0.dtype)

    # ---------------------------------------------------------------
    # Pass 2: subtract mean and store result
    # ---------------------------------------------------------------
    for start in range(0, S, step):
        for u in range(UNROLL):
            idx = start + offs + u * BLOCK_T
            mask = idx < S
            x_vals = tl.load(
                base_x + idx,
                mask=mask,
                other=0.0,
                cache_modifier='.cg',
            )
            y = x_vals - mean_cast  # scalar broadcast in registers
            tl.store(base_out + idx, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_T': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_T': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 256}, num_warps=4, num_stages=3),
    ],
    key=['D', 'H', 'W'],
)
@triton.jit
def subtract_spatial_mean_kernel_generic(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_T: tl.constexpr,
):
    """
    Generic-stride implementation:
        out = x - mean(x, dim=(2, 3, 4), keepdim=True)
    Works for any memory layout.
    """
    pid = tl.program_id(axis=0)
    NC = N * C
    if pid >= NC:
        return

    n = pid // C
    c = pid % C

    base_x = x_ptr + n * stride_xn + c * stride_xc
    base_out = out_ptr + n * stride_on + c * stride_oc

    S = D * H * W
    offs = tl.arange(0, BLOCK_T)

    # Pass 1: compute mean
    sum_val = tl.zeros((), dtype=tl.float32)
    for start in range(0, S, BLOCK_T):
        idx = start + offs
        mask = idx < S

        w = idx % W
        tmp = idx // W
        h = tmp % H
        d = tmp // H

        x_ptrs = base_x + d * stride_xd + h * stride_xh + w * stride_xw
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0, cache_modifier='.cg')
        x_vals_f32 = x_vals.to(tl.float32)
        sum_val += tl.sum(x_vals_f32, axis=0)

    mean = sum_val / S

    # Pre-cast mean once
    x0 = tl.load(base_x + 0 * stride_xd + 0 * stride_xh + 0 * stride_xw)
    mean_cast = tl.cast(mean, x0.dtype)

    # Pass 2: subtract mean and store
    for start in range(0, S, BLOCK_T):
        idx = start + offs
        mask = idx < S

        w = idx % W
        tmp = idx // W
        h = tmp % H
        d = tmp // H

        x_ptrs = base_x + d * stride_xd + h * stride_xh + w * stride_xw
        out_ptrs = base_out + d * stride_od + h * stride_oh + w * stride_ow

        x_vals = tl.load(x_ptrs, mask=mask, other=0.0, cache_modifier='.cg')
        y = x_vals - mean_cast
        tl.store(out_ptrs, y, mask=mask)


# ============================
# Wrapper Functions
# ============================

def subtract_spatial_mean_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Subtract spatial mean over (D, H, W) from each (N, C) slice of a 5D tensor:
        out = x - mean(x, dim=(2, 3, 4), keepdim=True)

    Fast path: contiguous [N, C, D, H, W] with simple [NC, S] layout.
    Fallback: generic-stride kernel for non-contiguous tensors.
    """
    assert x.ndim == 5, "Input must be a 5D tensor [N, C, D, H, W]"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        "Supported dtypes: float16, bfloat16, float32"

    N, C, D, H, W = x.shape
    out = torch.empty_like(x)

    S = D * H * W
    NC = N * C

    # Fast path: standard contiguous layout
    if x.is_contiguous(memory_format=torch.contiguous_format) and \
       x.stride(0) == C * S and x.stride(1) == S and \
       x.stride(2) == H * W and x.stride(3) == W and x.stride(4) == 1:

        grid = (NC,)
        subtract_spatial_mean_kernel_contiguous[grid](
            x, out,
            NC, S,
        )
    else:
        # Generic-stride kernel
        grid = (NC,)
        subtract_spatial_mean_kernel_generic[grid](
            x, out,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        )

    return out


# ============================
# PyTorch Module
# ============================

class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) + BatchNorm3d (PyTorch) +
    Triton-accelerated spatial mean subtraction.

    Semantics:
        x = conv_transpose(x)
        x = batch_norm(x)
        x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = subtract_spatial_mean_triton(x)
        return x
