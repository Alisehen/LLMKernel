import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
    ],
    key=["H_OUT", "W_OUT"],
)
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
    stride,
    padding,
    total_spatial,
    HAS_BIAS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < total_spatial

    h_out = hw_offsets // W_OUT
    w_out = hw_offsets % W_OUT

    h_in_origin = h_out * stride - padding
    w_in_origin = w_out * stride - padding

    base_nc = (n * C + c) * H * W
    out_base = (n * C + c) * total_spatial
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    w_base = c * KERNEL_SIZE * KERNEL_SIZE

    for kh in range(KERNEL_SIZE):
        h_in = h_in_origin + kh
        h_valid = (0 <= h_in) & (h_in < H)

        for kw in range(KERNEL_SIZE):
            w_in = w_in_origin + kw
            w_valid = (0 <= w_in) & (w_in < W)
            inner_mask = mask_hw & h_valid & w_valid

            in_offsets = base_nc + h_in * W + w_in
            vals = tl.load(x_ptr + in_offsets, mask=inner_mask, other=0.0)

            w_idx = w_base + kh * KERNEL_SIZE + kw
            weight = tl.load(w_ptr + w_idx)
            acc += vals * weight

    if HAS_BIAS:
        bias = tl.load(b_ptr + c)
        acc += bias

    out_offsets = out_base + hw_offsets
    tl.store(out_ptr + out_offsets, acc, mask=mask_hw)


def depthwise_conv2d_triton(x, weight, bias, stride, padding):
    x = x.contiguous()
    weight = weight.contiguous()

    N, C, H, W = x.shape
    KERNEL_SIZE = weight.shape[-1]
    H_OUT = (H + 2 * padding - KERNEL_SIZE) // stride + 1
    W_OUT = (W + 2 * padding - KERNEL_SIZE) // stride + 1
    total_spatial = H_OUT * W_OUT

    out = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)
    grid = lambda META: (N * C, triton.cdiv(total_spatial, META["BLOCK_HW"]))

    bias_ptr = bias if bias is not None else x.new_empty(1)

    depthwise_conv2d_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        N,
        C,
        H,
        W,
        H_OUT,
        W_OUT,
        stride,
        padding,
        total_spatial,
        HAS_BIAS=bias is not None,
        KERNEL_SIZE=KERNEL_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        weight = torch.empty(in_channels, 1, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            fan_in = kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(in_channels).uniform_(-bound, bound))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return depthwise_conv2d_triton(
            x,
            self.weight.view(self.in_channels, self.kernel_size, self.kernel_size),
            self.bias,
            self.stride,
            self.padding,
        )
