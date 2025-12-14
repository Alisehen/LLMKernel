import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_heightwise_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    K_H,
    STRIDE_H,
    STRIDE_W,
    PAD_H,
    PAD_W,
    DIL_H,
    DIL_W,
    H_OUT,
    W_OUT,
    stride_in_n,
    stride_in_c,
    stride_in_h,
    stride_in_w,
    stride_w_c,
    stride_w_kh,
    stride_out_n,
    stride_out_c,
    stride_out_h,
    stride_out_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_oh = tl.program_id(axis=1)
    pid_ow = tl.program_id(axis=2)

    NC = N * C
    if pid_nc >= NC or pid_oh >= H_OUT:
        return

    n = pid_nc // C
    c = pid_nc % C

    ow_start = pid_ow * BLOCK_W
    ow_offsets = ow_start + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < W_OUT

    base_in_ptr = x_ptr + n * stride_in_n + c * stride_in_c
    base_out_ptr = y_ptr + n * stride_out_n + c * stride_out_c

    in_w = ow_offsets * STRIDE_W - PAD_W
    valid_w = (in_w >= 0) & (in_w < W)

    if HAS_BIAS:
        bias_val = tl.load(b_ptr + c)
        acc = tl.full((BLOCK_W,), bias_val, tl.float32)
    else:
        acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    base_h = pid_oh * STRIDE_H - PAD_H
    w_base_ptr = w_ptr + c * stride_w_c

    for kh in range(K_H):
        in_h = base_h + kh * DIL_H
        row_ok = (in_h >= 0) and (in_h < H)
        if not row_ok:
            continue

        mask_k = mask_ow & valid_w
        ptrs = base_in_ptr + in_h * stride_in_h + in_w * stride_in_w
        vals = tl.load(ptrs, mask=mask_k, other=0.0)
        w_val = tl.load(w_base_ptr + kh * stride_w_kh)
        acc += vals * w_val

    out_ptrs = base_out_ptr + pid_oh * stride_out_h + ow_offsets * stride_out_w
    tl.store(out_ptrs, acc, mask=mask_ow)


def depthwise_conv_heightwise_triton(x, weight, bias, stride, padding, dilation):
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    N, C, H, W = x.shape
    K_H = weight.shape[2]
    K_W = weight.shape[3]
    assert K_W == 1, "Kernel width must be 1 for this specialized kernel."

    H_OUT = (H + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    W_OUT = (W + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    BLOCK_W = 64
    grid = (
        N * C,
        H_OUT,
        triton.cdiv(W_OUT, BLOCK_W),
    )

    depthwise_heightwise_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,  # placeholder pointer; unused when HAS_BIAS=False
        y,
        N,
        C,
        H,
        W,
        K_H,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        H_OUT,
        W_OUT,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        HAS_BIAS=bias is not None,
        BLOCK_W=BLOCK_W,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            fan_in = kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv_heightwise_triton(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
