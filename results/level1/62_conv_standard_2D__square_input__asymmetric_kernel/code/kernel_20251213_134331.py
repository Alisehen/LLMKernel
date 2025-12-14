# import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C_IN,
    H_IN,
    W_IN,
    C_OUT,
    H_OUT,
    W_OUT,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    groups,
    total_hw,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    C_IN_PER_G: tl.constexpr,
    C_OUT_PER_G: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_co = tl.program_id(1)

    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    hw_mask = hw_offsets < total_hw
    co_mask = co_offsets < C_OUT

    n_stride = H_OUT * W_OUT
    n = hw_offsets // n_stride
    rem = hw_offsets % n_stride
    oh = rem // W_OUT
    ow = rem % W_OUT

    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    group_id = co_offsets // C_OUT_PER_G
    cin_base = group_id * C_IN_PER_G

    oh_f32 = oh.to(tl.int32)
    ow_f32 = ow.to(tl.int32)

    for kh in tl.static(range(K_H)):
        ih = oh_f32 * stride_h - pad_h + kh * dil_h
        valid_h = (ih >= 0) & (ih < H_IN)
        ih_safe = tl.where(valid_h, ih, 0)
        for kw in tl.static(range(K_W)):
            iw = ow_f32 * stride_w - pad_w + kw * dil_w
            valid_w = (iw >= 0) & (iw < W_IN)
            valid_hw = hw_mask & valid_h & valid_w
            iw_safe = tl.where(valid_w, iw, 0)

            ih_mat = ih_safe[:, None]
            iw_mat = iw_safe[:, None]
            n_mat = n[:, None]

            for ci in tl.static(range(C_IN_PER_G)):
                in_c = cin_base + ci
                x_idx = (((n_mat * C_IN) + in_c[None, :]) * H_IN + ih_mat) * W_IN + iw_mat
                x_mask = valid_hw[:, None] & co_mask[None, :]
                x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0).to(tl.float32)

                w_idx = (((co_offsets * C_IN_PER_G) + ci) * K_H + kh) * K_W + kw
                w_vals = tl.load(w_ptr + w_idx, mask=co_mask, other=0.0).to(tl.float32)
                acc += x_vals * w_vals[None, :]

    out_idx = (((n[:, None] * C_OUT) + co_offsets[None, :]) * H_OUT + oh[:, None]) * W_OUT + ow[:, None]
    out_mask = hw_mask[:, None] & co_mask[None, :]
    tl.store(out_ptr + out_idx, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_IN, H_IN, W_IN = x.shape
    C_OUT, _, K_H, K_W = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_OUT = (H_IN + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_OUT = (W_IN + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    out = torch.empty((N, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_HW = 64
    BLOCK_CO = 64

    grid = (
        triton.cdiv(N * H_OUT * W_OUT, BLOCK_HW),
        triton.cdiv(C_OUT, BLOCK_CO),
    )

    conv2d_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        N,
        C_IN,
        H_IN,
        W_IN,
        C_OUT,
        H_OUT,
        W_OUT,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        N * H_OUT * W_OUT,
        HAS_BIAS=bias is not None,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CO=BLOCK_CO,
        K_H=K_H,
        K_W=K_W,
        C_IN_PER_G=weight.shape[1],
        C_OUT_PER_G=C_OUT // groups,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
