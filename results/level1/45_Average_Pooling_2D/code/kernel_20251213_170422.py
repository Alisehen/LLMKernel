# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=3),
    ],
    key=["N", "C", "H", "W", "OH", "OW"],
)
@triton.jit
def avgpool2d_kernel(
    x_ptr,  # *f32, [N, C, H, W] contiguous
    y_ptr,  # *f32, [N, C, OH, OW] contiguous
    N, C, H, W,
    OH, OW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # ---- program ids ----
    pid_nc_oh = tl.program_id(axis=0)  # spans N * C * OH
    pid_w_blk = tl.program_id(axis=1)  # spans OW blocks

    # ---- decode pid_nc_oh -> (n, c, oh) ----
    oh = pid_nc_oh % OH
    tmp = pid_nc_oh // OH
    c = tmp % C
    n = tmp // C

    # ---- output W indices for this block ----
    offs_w = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = offs_w < OW
    tl.multiple_of(offs_w, BLOCK_W)
    tl.max_contiguous(offs_w, BLOCK_W)

    # ---- precompute strides (contiguous NCHW) ----
    HW = H * W
    sC = HW
    sN = C * HW

    # base offset for (n, c, 0, 0)
    base_nc = n * sN + c * sC

    # accumulator
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    stride_h = STRIDE_H
    stride_w = STRIDE_W
    pad_h = PAD_H
    pad_w = PAD_W

    # base input h coordinate for this output row
    base_ih = oh * stride_h - pad_h

    # base iw for all lanes (vector)
    base_iw = offs_w * stride_w - pad_w

    # loop over kernel window (unrolled by Triton)
    for kh in tl.static_range(KERNEL_H):
        ih = base_ih + kh  # scalar
        valid_h = (ih >= 0) & (ih < H)
        # ensure row_offset always in-bounds even when masked
        ih_safe = tl.where(valid_h, ih, 0)
        row_offset = base_nc + ih_safe * W  # scalar

        for kw in tl.static_range(KERNEL_W):
            iw = base_iw + kw  # vector
            valid_w = (iw >= 0) & (iw < W)
            mask = mask_ow & valid_h & valid_w

            # ensure column indices are in-bounds even for masked lanes
            iw_safe = tl.where(valid_w, iw, 0)
            ptrs = x_ptr + row_offset + iw_safe
            vals = tl.load(ptrs, mask=mask, other=0.0)
            acc += vals

    # average (count_include_pad = True)
    denom = KERNEL_H * KERNEL_W
    acc *= 1.0 / denom

    # ---- write output ----
    OH_OW = OH * OW
    sC_out = OH_OW
    sN_out = C * OH_OW

    base_nc_out = n * sN_out + c * sC_out
    row_out_offset = base_nc_out + oh * OW

    # keep output pointers in-bounds for all lanes
    offs_w_safe = tl.where(mask_ow, offs_w, 0)
    out_ptrs = y_ptr + row_out_offset + offs_w_safe
    tl.store(out_ptrs, acc, mask=mask_ow)


def triton_avg_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
) -> torch.Tensor:
    """
    Average Pooling 2D using Triton (NCHW, count_include_pad=True, ceil_mode=False).
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "This implementation currently supports float32 only"
    assert x.dim() == 4, "Input must be 4D NCHW"

    if stride is None:
        stride = kernel_size

    N, C, H, W = x.shape
    kh = kw = int(kernel_size)
    sh = sw = int(stride)
    ph = pw = int(padding)

    # PyTorch AvgPool2d output size (ceil_mode=False)
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    def grid(meta):
        block_w = meta["BLOCK_W"]
        return (
            N * C * OH,               # pid_nc_oh
            triton.cdiv(OW, block_w), # pid_w_blk
        )

    avgpool2d_kernel[grid](
        x,
        y,
        N,
        C,
        H,
        W,
        OH,
        OW,
        sh,
        sw,
        ph,
        pw,
        KERNEL_H=kh,
        KERNEL_W=kw,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-backed 2D Average Pooling (NCHW) matching nn.AvgPool2d
    for kernel_size, stride, padding, count_include_pad=True, ceil_mode=False.
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
