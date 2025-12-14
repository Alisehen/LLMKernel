import torch
import torch.nn as nn
import triton
import triton.language as tl

def _to_triton_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    if torch_dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype: {torch_dtype}")

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    C_in,
    H_in,
    W_in,
    C_out,
    OH,
    OW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    channels_per_group,
    HAS_BIAS: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    total_per_n = C_out * OH
    n = pid_row // total_per_n
    rem = pid_row % total_per_n
    oc = rem // OH
    oh = rem % OH

    ow_offsets = pid_col * BLOCK_W + tl.arange(0, BLOCK_W)
    valid_ow = ow_offsets < OW

    oh_in = oh * stride_h - pad_h
    ow_in_base = ow_offsets * stride_w - pad_w

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    c_in = oc // channels_per_group
    base_in_c = (n * C_in + c_in) * H_in

    for kh in range(KH):
        in_y = oh_in + kh
        mask_y = (in_y >= 0) & (in_y < H_in)
        y_offset = (base_in_c + in_y) * W_in

        for kw in range(KW):
            in_x = ow_in_base + kw
            mask_x = (in_x >= 0) & (in_x < W_in)
            mask = valid_ow & mask_x & mask_y

            ptrs = y_offset + in_x
            ptrs = tl.where(mask, ptrs, 0)
            vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)

            w_offset = ((oc * KH) + kh) * KW + kw
            weight = tl.load(w_ptr + w_offset)

            acc += vals.to(tl.float32) * weight.to(tl.float32)

    if HAS_BIAS:
        bias_val = tl.load(b_ptr + oc)
        acc += bias_val.to(tl.float32)

    out_vals = acc.to(OUTPUT_DTYPE)
    out_ptrs = (((n * C_out + oc) * OH) + oh) * OW + ow_offsets
    tl.store(out_ptr + out_ptrs, out_vals, mask=valid_ow)

def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == weight.dtype, "Input and weight dtypes must match"
    stride_h = stride_w = stride
    pad_h = pad_w = padding

    B, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    kernel_size = weight.shape[2]
    channels_per_group = C_out // C_in

    OH = (H_in + 2 * pad_h - kernel_size) // stride_h + 1
    OW = (W_in + 2 * pad_w - kernel_size) // stride_w + 1

    output = torch.empty((B, C_out, OH, OW), device=x.device, dtype=x.dtype)

    bias_ptr = bias if bias is not None else output
    has_bias = bias is not None

    BLOCK_W = 128
    grid = (
        B * C_out * OH,
        triton.cdiv(OW, BLOCK_W),
    )

    depthwise_conv2d_kernel[grid](
        x,
        weight,
        bias_ptr,
        output,
        B,
        C_in,
        H_in,
        W_in,
        C_out,
        OH,
        OW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        channels_per_group,
        HAS_BIAS=has_bias,
        KH=kernel_size,
        KW=kernel_size,
        BLOCK_W=BLOCK_W,
        OUTPUT_DTYPE=_to_triton_dtype(x.dtype),
    )
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding)
