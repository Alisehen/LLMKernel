# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_NC": 16, "BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_NC": 32, "BLOCK_HW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_NC": 32, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_NC": 64, "BLOCK_HW": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_NC": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
    ],
    key=["N", "C", "H", "W", "H_OUT", "W_OUT", "stride", "padding", "KERNEL_SIZE"],
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
    stride: tl.constexpr,
    padding: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    nc_offsets = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    total_nc = N * C
    total_hw = H_OUT * W_OUT

    mask_nc = nc_offsets < total_nc
    mask_hw = hw_offsets < total_hw
    valid_mask = mask_nc[:, None] & mask_hw[None, :]

    nc_i64 = nc_offsets.to(tl.int64)
    hw_i64 = hw_offsets.to(tl.int64)

    h_out = hw_i64 // W_OUT
    w_out = hw_i64 % W_OUT

    h_in_base = h_out * stride - padding
    w_in_base = w_out * stride - padding

    c_idx = tl.mod(nc_offsets, C)
    weight_base = c_idx * (KERNEL_SIZE * KERNEL_SIZE)

    acc = tl.zeros((BLOCK_NC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(KERNEL_SIZE):
        h_in = h_in_base + kh
        h_in_valid = (h_in >= 0) & (h_in < H)
        h_in_safe = tl.where(h_in_valid, h_in, 0)
        h_in_i64 = h_in_safe.to(tl.int64)[None, :]

        for kw in tl.static_range(KERNEL_SIZE):
            w_in = w_in_base + kw
            w_in_valid = (w_in >= 0) & (w_in < W)
            w_in_safe = tl.where(w_in_valid, w_in, 0)
            w_in_i64 = w_in_safe.to(tl.int64)[None, :]

            load_mask = valid_mask & h_in_valid[None, :] & w_in_valid[None, :]

            input_offset = ((nc_i64[:, None] * H) + h_in_i64) * W + w_in_i64
            vals = tl.load(x_ptr + input_offset, mask=load_mask, other=0.0)
            vals = vals.to(tl.float32)

            w_offsets = weight_base + kh * KERNEL_SIZE + kw
            weights = tl.load(w_ptr + w_offsets, mask=mask_nc, other=0.0)[:, None]
            weights = weights.to(tl.float32)

            acc += vals * weights

    if HAS_BIAS:
        bias = tl.load(b_ptr + c_idx, mask=mask_nc, other=0.0)[:, None]
        acc += bias

    acc = acc.to(out_ptr.dtype.element_ty)
    out_offset = ((nc_i64[:, None] * H_OUT) + h_out[None, :]) * W_OUT + w_out[None, :]
    tl.store(out_ptr + out_offset, acc, mask=valid_mask)


def depthwise_conv2d_triton(x, weight, bias, stride, padding):
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C, H, W = x.shape
    K = weight.shape[-1]
    H_OUT = (H + 2 * padding - K) // stride + 1
    W_OUT = (W + 2 * padding - K) // stride + 1

    out = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * C, META["BLOCK_NC"]),
        triton.cdiv(H_OUT * W_OUT, META["BLOCK_HW"]),
    )

    bias_ptr = bias if bias is not None else x.new_empty(1, dtype=x.dtype)

    depthwise_conv2d_kernel[grid](
        x,
        weight.view(C, K * K),
        bias_ptr,
        out,
        N,
        C,
        H,
        W,
        H_OUT,
        W_OUT,
        stride=stride,
        padding=padding,
        HAS_BIAS=bias is not None,
        KERNEL_SIZE=K,
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
