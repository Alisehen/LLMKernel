# import section
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_HW": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=3),
    ],
    key=["H_out", "W_out"],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    KH,
    KW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    H_out,
    W_out,
    HW_out,
    has_bias,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    n = pid_nc // C
    c = pid_nc % C

    hw_start = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < HW_out

    oh = hw_offsets // W_out
    ow = hw_offsets % W_out

    base_x = ((n * C + c) * H) * W
    base_y = ((n * C + c) * H_out) * W_out

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for kh in range(0, KH):
        ih = oh * stride_h - pad_h + kh * dil_h
        mask_h = (ih >= 0) & (ih < H)
        for kw in range(0, KW):
            iw = ow * stride_w - pad_w + kw * dil_w
            mask_w = (iw >= 0) & (iw < W)
            mask = mask_hw & mask_h & mask_w

            x_idx = base_x + ih * W + iw
            w_idx = (c * KH + kh) * KW + kw

            x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_idx).to(tl.float32)
            acc += x_val * w_val

    if has_bias:
        bias_val = tl.load(bias_ptr + c).to(tl.float32)
        acc += bias_val

    y_ptrs = y_ptr + base_y + hw_offsets
    tl.store(y_ptrs, acc.to(OUTPUT_DTYPE), mask=mask_hw)


def triton_depthwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    N, C, H, W = x.shape
    KH, KW = weight.shape[-2:]
    H_out = (H + 2 * padding_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (KW - 1) - 1) // stride_w + 1
    HW_out = H_out * W_out

    y = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

    has_bias = 1 if bias is not None else 0
    bias_ptr = bias if bias is not None else weight

    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    w_flat = weight.contiguous().view(-1)

    output_dtype = tl.float16 if x.dtype == torch.float16 else tl.float32

    grid = lambda meta: (N * C, triton.cdiv(HW_out, meta["BLOCK_HW"]))

    depthwise_conv2d_kernel[grid](
        x_flat,
        w_flat,
        bias_ptr,
        y_flat,
        N,
        C,
        H,
        W,
        KH,
        KW,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        H_out,
        W_out,
        HW_out,
        has_bias,
        OUTPUT_DTYPE=output_dtype,
    )

    return y


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert in_channels == out_channels
        assert groups == in_channels
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.weight = nn.Parameter(
            torch.empty(out_channels, 1, kernel_size_h, kernel_size_w)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = kernel_size_h * kernel_size_w
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.view(self.weight.shape[0], self.weight.shape[2], self.weight.shape[3])
        return triton_depthwise_conv2d(
            x,
            weight,
            self.bias,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )
