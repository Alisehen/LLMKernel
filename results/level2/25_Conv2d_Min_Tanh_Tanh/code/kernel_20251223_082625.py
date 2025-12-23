import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_tanh_tanh_kernel(
    x_ptr, y_ptr,
    B, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_C: tl.constexpr,
):
    # Program IDs for batch, height, width
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Guard against out-of-bounds in case grid is larger than tensor dims
    mask_n = pid_n < B
    mask_h = pid_h < H
    mask_w = pid_w < W
    in_bounds = mask_n & mask_h & mask_w

    # If out of bounds, we still must not early-return; use masking
    # Base pointer for this (n, h, w) location in the input tensor
    x_base_offset = (
        pid_n * stride_xn
        + pid_h * stride_xh
        + pid_w * stride_xw
    )
    x_base_ptr = x_ptr + x_base_offset

    # Initialize running minimum (in float32 for numeric stability)
    min_val = tl.full((), float("inf"), tl.float32)

    # Reduce over channel dimension C in blocks of BLOCK_C
    # Each iteration handles up to BLOCK_C channels
    c = 0
    while c < C:
        offs_c = c + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        # Only load if the (n,h,w) position is in bounds
        mask = mask_c & in_bounds

        x_ptrs = x_base_ptr + offs_c * stride_xc
        x_vals = tl.load(x_ptrs, mask=mask, other=float("inf"))

        # Accumulate minimum
        block_min = tl.min(x_vals, axis=0)
        min_val = tl.minimum(min_val, block_min)

        c += BLOCK_C

    # Apply two successive tanh operations:
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # First tanh
    t = 2.0 * min_val
    e = tl.exp(t)
    tanh1 = (e - 1.0) / (e + 1.0)
    # Second tanh
    t2 = 2.0 * tanh1
    e2 = tl.exp(t2)
    out_val = (e2 - 1.0) / (e2 + 1.0)

    # Store result to output at channel index 0 (keepdim=True)
    y_offset = (
        pid_n * stride_yn
        + 0 * stride_yc  # channel is always 0 due to keepdim=True
        + pid_h * stride_yh
        + pid_w * stride_yw
    )

    # Only store if within bounds
    tl.store(y_ptr + y_offset, out_val, mask=in_bounds)


def min_tanh_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Fused replacement for:
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = torch.tanh(x)
        x = torch.tanh(x)
    Assumes x is NCHW and float32 on CUDA.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Kernel currently assumes float32"

    B, C, H, W = x.shape
    y = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (B, H, W)

    min_tanh_tanh_kernel[grid](
        x, y,
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_C=64,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the original model using Triton for:
      min over channels + tanh + tanh
    Convolution is still performed via cuDNN (nn.Conv2d).
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = min_tanh_tanh(x)
        return x
