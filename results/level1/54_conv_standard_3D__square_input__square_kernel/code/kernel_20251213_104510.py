# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    Cin,
    Cout,
    D,
    H,
    W,
    KD,
    KH,
    KW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    D_out,
    H_out,
    W_out,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = N * Cout * D_out * H_out * W_out
    mask = offsets < total

    ow = offsets % W_out
    tmp = offsets // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    tmp = tmp // D_out
    oc = tmp % Cout
    n = tmp // Cout

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + oc, mask=mask, other=0.0).to(tl.float32)
        acc += bias_vals

    for ic in range(0, Cin):
        for kd in range(0, KD):
            in_d = od * stride_d + kd * dil_d - pad_d
            mask_d = (in_d >= 0) & (in_d < D)
            for kh in range(0, KH):
                in_h = oh * stride_h + kh * dil_h - pad_h
                mask_h = (in_h >= 0) & (in_h < H)
                for kw in range(0, KW):
                    in_w = ow * stride_w + kw * dil_w - pad_w
                    mask_w = (in_w >= 0) & (in_w < W)
                    valid = mask & mask_d & mask_h & mask_w

                    w_offsets = (
                        (((oc * Cin) + ic) * KD + kd) * KH + kh
                    ) * KW + kw
                    w_vals = tl.load(weight_ptr + w_offsets, mask=mask, other=0.0).to(tl.float32)

                    inp_offsets = (
                        ((((n * Cin) + ic) * D + in_d) * H + in_h) * W + in_w
                    )
                    inp_vals = tl.load(input_ptr + inp_offsets, mask=valid, other=0.0).to(tl.float32)

                    acc += inp_vals * w_vals

    tl.store(output_ptr + offsets, acc, mask=mask)


def _triple(x):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    return (x, x, x)


def triton_conv3d(x, weight, bias, stride, padding, dilation):
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise NotImplementedError("Only float32 tensors are supported in this implementation.")
    stride_d, stride_h, stride_w = _triple(stride)
    pad_d, pad_h, pad_w = _triple(padding)
    dil_d, dil_h, dil_w = _triple(dilation)

    N, Cin, D, H, W = x.shape
    Cout, Cin_w, KD, KH, KW = weight.shape
    if Cin != Cin_w:
        raise NotImplementedError("Grouped convolution is not supported in this implementation.")

    D_out = (D + 2 * pad_d - dil_d * (KD - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    output = torch.empty((N, Cout, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    total = output.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    bias_ptr = bias.contiguous() if bias is not None else output.new_empty(1)

    conv3d_kernel[grid](
        x_contig,
        w_contig,
        bias_ptr,
        output,
        N,
        Cin,
        Cout,
        D,
        H,
        W,
        KD,
        KH,
        KW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        D_out,
        H_out,
        W_out,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("Grouped convolution is not supported in this implementation.")
        kernel_size = _triple(kernel_size) if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)
