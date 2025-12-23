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
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)   # batch dimension
    pid_hw = tl.program_id(1)  # flattened H*W tiles

    # Offsets along flattened spatial dimension
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    HW = H * W

    # Masks
    mask_n = pid_n < B
    mask_hw = offs_hw < HW
    mask_valid = mask_hw & mask_n

    # Recover (h, w) from flattened index
    h = offs_hw // W
    w = offs_hw - h * W

    # Base pointer offsets for (n, h, w) with c = 0
    base_x = (
        pid_n * stride_xn
        + h * stride_xh
        + w * stride_xw
    )  # [BLOCK_HW]

    # Initialize running minimum over channels for each (n, h, w)
    min_val = tl.full((BLOCK_HW,), float("inf"), tl.float32)

    # Loop over channels in blocks of BLOCK_C
    c = 0
    while c < C:
        offs_c = c + tl.arange(0, BLOCK_C)      # [BLOCK_C]
        mask_c = offs_c < C                     # [BLOCK_C]

        # Pointers: [BLOCK_C, BLOCK_HW]
        ptrs = x_ptr + base_x[None, :] + offs_c[:, None] * stride_xc
        load_mask = mask_c[:, None] & mask_valid[None, :]

        x_vals = tl.load(ptrs, mask=load_mask, other=float("inf"))  # [BLOCK_C, BLOCK_HW]

        # Reduce along channel axis (axis=0)
        block_min = tl.min(x_vals, axis=0)  # [BLOCK_HW]
        min_val = tl.minimum(min_val, block_min)

        c += BLOCK_C

    # Apply tanh twice: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    t1 = 2.0 * min_val
    e1 = tl.exp(t1)
    tanh1 = (e1 - 1.0) / (e1 + 1.0)

    t2 = 2.0 * tanh1
    e2 = tl.exp(t2)
    out_val = (e2 - 1.0) / (e2 + 1.0)

    # Output pointers (channel is 0, because keepdim=True)
    base_y = (
        pid_n * stride_yn
        + h * stride_yh
        + w * stride_yw
    )  # [BLOCK_HW]

    tl.store(y_ptr + base_y, out_val, mask=mask_valid)


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

    def grid(meta):
        BLOCK_HW = meta["BLOCK_HW"]
        return (
            B,
            triton.cdiv(H * W, BLOCK_HW),
        )

    min_tanh_tanh_kernel[grid](
        x, y,
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_HW=256,   # power-of-2
        BLOCK_C=64,     # power-of-2
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized version of the original model using Triton for:
      min over channels + tanh + tanh.
    Convolution is still performed via cuDNN (nn.Conv2d).
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = min_tanh_tanh(x)
        return x
