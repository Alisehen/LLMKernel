# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8,  'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 4,  'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 8,  'BLOCK_W': 64}, num_warps=8, num_stages=2),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def fused_maxpool_hardtanh_mean_tanh_kernel(
    x_ptr, out_ptr,
    N, C,
    H, W,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc,
    sH, sW,
    hardtanh_min, hardtanh_max,
    K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    """
    Fully fused kernel:
        MaxPool2d + Hardtanh + GlobalMean (over pooled H_out, W_out) + tanh

    Input:  x[N, C, H, W]
    Output: out[N, C, 1, 1]
    """
    pid_nc = tl.program_id(0)
    total_nc = N * C
    if pid_nc >= total_nc:
        return

    n = pid_nc // C
    c = pid_nc % C

    x_nc_ptr = x_ptr + n * stride_xn + c * stride_xc
    out_nc_ptr = out_ptr + n * stride_on + c * stride_oc

    # Accumulate global mean in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Cast hardtanh bounds to fp32 once
    ht_min = hardtanh_min.to(tl.float32)
    ht_max = hardtanh_max.to(tl.float32)

    # Loop over output tiles in (H_out, W_out)
    # Tile shape: [BLOCK_H, BLOCK_W]
    for oh_start in range(0, H_out, BLOCK_H):
        offs_oh = oh_start + tl.arange(0, BLOCK_H)[:, None]
        for ow_start in range(0, W_out, BLOCK_W):
            offs_ow = ow_start + tl.arange(0, BLOCK_W)[None, :]

            # Mask for valid pooled output coords
            mask_hw = (offs_oh < H_out) & (offs_ow < W_out)

            # Map pooled output indices -> input coordinates
            base_iy = offs_oh * sH
            base_ix = offs_ow * sW

            # Max-pooling window over input, accumulated in fp32
            max_vals = tl.full((BLOCK_H, BLOCK_W), -float("inf"), dtype=tl.float32)

            for kh in range(0, K_H):
                iy = base_iy + kh
                for kw in range(0, K_W):
                    ix = base_ix + kw
                    x_ptrs = x_nc_ptr + iy * stride_xh + ix * stride_xw
                    vals = tl.load(x_ptrs, mask=mask_hw, other=-float("inf"))
                    vals_f32 = vals.to(tl.float32)
                    max_vals = tl.maximum(max_vals, vals_f32)

            # Fused Hardtanh
            max_vals = tl.minimum(tl.maximum(max_vals, ht_min), ht_max)

            # Zero-out contributions from invalid lanes (masked outputs)
            max_vals = tl.where(mask_hw, max_vals, 0.0)

            # Accumulate sum over this tile into scalar acc
            tile_sum = tl.sum(max_vals, axis=0)
            tile_sum = tl.sum(tile_sum, axis=0)
            acc += tile_sum

    # Global mean over all pooled outputs
    HW_out = H_out * W_out
    mean_val = acc / HW_out

    # tanh(mean) using exp: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    two_mean = 2.0 * mean_val
    e2x = tl.exp(two_mean)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)

    # Single final store per (N, C)
    tl.store(out_nc_ptr, tanh_val)


def fused_post_convtranspose(
    x,
    maxpool_kernel_size,
    maxpool_stride,
    hardtanh_min,
    hardtanh_max,
):
    """
    Fused:
        MaxPool2d + Hardtanh + GlobalMean (H_out,W_out, keepdim=True) + tanh

    x: [N, C, H, W] (output of ConvTranspose2d)
    returns: [N, C, 1, 1]
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."

    N, C, H, W = x.shape

    # Normalize kernel_size and stride to (kH, kW), (sH, sW)
    if isinstance(maxpool_kernel_size, int):
        kH = kW = maxpool_kernel_size
    else:
        kH, kW = maxpool_kernel_size

    if isinstance(maxpool_stride, int):
        sH = sW = maxpool_stride
    else:
        sH, sW = maxpool_stride

    # Compute pooled output spatial size (no padding)
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    # Final output [N, C, 1, 1]
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    grid = lambda META: (N * C,)

    fused_maxpool_hardtanh_mean_tanh_kernel[grid](
        x, out,
        N, C,
        H, W,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1),
        sH, sW,
        float(hardtanh_min), float(hardtanh_max),
        K_H=kH, K_W=kW,
    )

    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) +
    single fused Triton kernel implementing:
        MaxPool2d + Hardtanh + GlobalMean (H,W, keepdim=True) + tanh
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        maxpool_kernel_size,
        maxpool_stride,
        hardtanh_min,
        hardtanh_max,
    ):
        super(ModelNew, self).__init__()
        # ConvTranspose2d stays as PyTorch native
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # Store pooling / activation parameters for Triton part
        if isinstance(maxpool_kernel_size, int):
            self.maxpool_kernel_size = (maxpool_kernel_size, maxpool_kernel_size)
        else:
            self.maxpool_kernel_size = tuple(maxpool_kernel_size)

        if isinstance(maxpool_stride, int):
            self.maxpool_stride = (maxpool_stride, maxpool_stride)
        else:
            self.maxpool_stride = tuple(maxpool_stride)

        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        # ConvTranspose2d in PyTorch
        x = self.conv_transpose(x)
        # Fully fused Triton kernel: MaxPool2d + Hardtanh + Mean + tanh
        x = fused_post_convtranspose(
            x,
            self.maxpool_kernel_size,
            self.maxpool_stride,
            self.hardtanh_min,
            self.hardtanh_max,
        )
        return x
