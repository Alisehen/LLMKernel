# optimized Triton code
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
    ],
    key=['HW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    B,
    IC,
    IH,
    IW,
    OC,
    KH,
    KW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    OH,
    OW,
    HW,
    has_bias: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_ocb = tl.program_id(0)
    pid_hw = tl.program_id(1)

    oc = pid_ocb % OC
    b = pid_ocb // OC

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < HW

    oh = hw_offsets // OW
    ow = hw_offsets % OW

    ih_base = oh * stride_h - pad_h
    iw_base = ow * stride_w - pad_w

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for ic in range(0, IC):
        base_x_offset = ((b * IC + ic) * IH) * IW
        w_ic_base = ((oc * IC + ic) * KH) * KW
        for kh in range(0, KH):
            ih = ih_base + kh * dil_h
            valid_h = (ih >= 0) & (ih < IH)
            ih_clamped = tl.where(valid_h, ih, 0)
            row_offset = base_x_offset + ih_clamped * IW
            w_kh_base = w_ic_base + kh * KW
            for kw in range(0, KW):
                iw = iw_base + kw * dil_w
                valid_w = (iw >= 0) & (iw < IW)
                valid = mask_hw & valid_h & valid_w
                iw_clamped = tl.where(valid_w, iw, 0)
                x_offset = row_offset + iw_clamped
                w_offset = w_kh_base + kw

                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    if has_bias:
        bias = tl.load(b_ptr + oc)
        acc += bias

    y_offset = (((b * OC + oc) * OH) + oh) * OW + ow
    tl.store(y_ptr + y_offset, acc, mask=mask_hw)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple) -> torch.Tensor:
    assert x.is_contiguous()
    assert weight.is_contiguous()

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    B, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape

    OH = (IH + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (IW + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    HW = OH * OW

    y = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x.new_empty(1)

    grid = lambda meta: (B * OC, triton.cdiv(HW, meta['BLOCK_HW']))

    conv2d_kernel[grid](
        x,
        weight,
        bias_ptr,
        y,
        B,
        IC,
        IH,
        IW,
        OC,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        OH,
        OW,
        HW,
        has_bias=has_bias,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
