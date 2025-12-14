import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    H_OUT,
    W_OUT,
    K_H,
    K_W,
    STRIDE_H,
    STRIDE_W,
    PAD_H,
    PAD_W,
    DIL_H,
    DIL_W,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    if pid_nc >= N * C:
        return

    hw_total = H_OUT * W_OUT
    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < hw_total

    w_out_idx = hw_offsets % W_OUT
    h_out_idx = hw_offsets // W_OUT

    n = pid_nc // C
    c = pid_nc % C

    base_in_nc = (n * C + c) * H * W
    base_out_nc = (n * C + c) * H_OUT * W_OUT

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for kh in range(K_H):
        in_h = h_out_idx * STRIDE_H + kh * DIL_H - PAD_H
        valid_h = (in_h >= 0) & (in_h < H)
        for kw in range(K_W):
            in_w = w_out_idx * STRIDE_W + kw * DIL_W - PAD_W
            valid_w = (in_w >= 0) & (in_w < W)
            mask = mask_hw & valid_h & valid_w

            input_offsets = base_in_nc + in_h * W + in_w
            vals = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
            vals = vals.to(tl.float32)

            w_idx = c * K_H * K_W + kh * K_W + kw
            weight = tl.load(w_ptr + w_idx).to(tl.float32)

            acc += vals * weight

    if HAS_BIAS:
        bias = tl.load(b_ptr + c).to(tl.float32)
        acc += bias

    tl.store(out_ptr + base_out_nc + hw_offsets, acc, mask=mask_hw)


@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    C_IN,
    C_OUT,
    H,
    W,
    HAS_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_oc = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    hw_total = N * H * W
    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < hw_total

    HW = H * W
    plane_offsets = hw_offsets % HW
    n_offsets = hw_offsets // HW

    for oc_inner in tl.static_range(BLOCK_OC):
        oc = pid_oc * BLOCK_OC + oc_inner
        if oc >= C_OUT:
            continue

        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
        for c in range(C_IN):
            weight = tl.load(w_ptr + oc * C_IN + c).to(tl.float32)
            input_offsets = (n_offsets * C_IN + c) * HW + plane_offsets
            vals = tl.load(x_ptr + input_offsets, mask=mask_hw, other=0.0).to(tl.float32)
            acc += vals * weight

        if HAS_BIAS:
            bias = tl.load(b_ptr + oc).to(tl.float32)
            acc += bias

        out_offsets = (n_offsets * C_OUT + oc) * HW + plane_offsets
        tl.store(out_ptr + out_offsets, acc, mask=mask_hw)


def depthwise_conv2d_triton(x, weight, bias, stride, padding, dilation):
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else padding
    dil_h, dil_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    N, C, H, W = x.shape
    K_H, K_W = weight.shape[-2], weight.shape[-1]

    H_OUT = (H + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_OUT = (W + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    out = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = (
        N * C,
        triton.cdiv(H_OUT * W_OUT, 128),
    )

    depthwise_conv2d_kernel[grid](
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous() if bias is not None else weight,
        out,
        N,
        C,
        H,
        W,
        H_OUT,
        W_OUT,
        K_H,
        K_W,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        HAS_BIAS=int(bias is not None),
        BLOCK_HW=128,
    )

    return out


def pointwise_conv2d_triton(x, weight, bias):
    N, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]

    weight_2d = weight.view(C_OUT, C_IN)

    out = torch.empty((N, C_OUT, H, W), device=x.device, dtype=x.dtype)

    grid = (
        triton.cdiv(C_OUT, 4),
        triton.cdiv(N * H * W, 128),
    )

    pointwise_conv2d_kernel[grid](
        x.contiguous(),
        weight_2d.contiguous(),
        bias.contiguous() if bias is not None else weight_2d,
        out,
        N,
        C_IN,
        C_OUT,
        H,
        W,
        HAS_BIAS=int(bias is not None),
        BLOCK_OC=4,
        BLOCK_HW=128,
    )

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = depthwise_conv2d_triton(
            x,
            self.depthwise.weight,
            self.depthwise.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        x = pointwise_conv2d_triton(
            x,
            self.pointwise.weight,
            self.pointwise.bias,
        )
        return x
