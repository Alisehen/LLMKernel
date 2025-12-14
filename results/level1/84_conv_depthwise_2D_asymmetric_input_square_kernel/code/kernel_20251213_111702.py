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

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 96}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 128}, num_warps=8, num_stages=3),
    ],
    key=["OW"],
)
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
    mask_w = ow_offsets < OW

    oh_in = oh * stride_h - pad_h
    ow_in = ow_offsets * stride_w - pad_w

    c_in = oc // channels_per_group
    base_nc = (n * C_in + c_in) * H_in

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    weight_base = oc * KH * KW
    weight_idx = tl.arange(0, KH * KW)
    weights = tl.load(w_ptr + weight_base + weight_idx)

    for kh in range(KH):
        in_y = oh_in + kh
        mask_y = (in_y >= 0) & (in_y < H_in)
        row_base = tl.where(mask_y, (base_nc + in_y) * W_in, 0)

        for kw in range(KW):
            in_x = ow_in + kw
            mask_x = (in_x >= 0) & (in_x < W_in)
            mask = mask_w & mask_x & mask_y

            ptrs = row_base + in_x
            vals = tl.load(x_ptr + ptrs, mask=mask, other=0.0)

            w_val = weights[kh * KW + kw]
            acc += vals.to(tl.float32) * w_val.to(tl.float32)

    if HAS_BIAS:
        bias_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += bias_val

    out_vals = acc.to(OUTPUT_DTYPE)
    out_ptrs = (((n * C_out + oc) * OH) + oh) * OW + ow_offsets
    tl.store(out_ptr + out_ptrs, out_vals, mask=mask_w)

def triton_depthwise_conv2d(x, weight, bias, stride, padding):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == weight.dtype, "Input and weight dtypes must match"

    stride_h = stride_w = stride
    pad_h = pad_w = padding

    B, C_in, H_in, W_in = x.shape
    C_out = weight.shape[0]
    kernel_size = weight.shape[2]
    channels_per_group = max(C_out // C_in, 1)

    OH = (H_in + 2 * pad_h - kernel_size) // stride_h + 1
    OW = (W_in + 2 * pad_w - kernel_size) // stride_w + 1

    output = torch.empty((B, C_out, OH, OW), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    bias_ptr = bias if has_bias else output

    grid = lambda META: (
        B * C_out * OH,
        triton.cdiv(OW, META["BLOCK_W"]),
    )

    depthwise_conv2d_kernel[grid](
        x,
        weight.reshape(C_out, kernel_size * kernel_size),
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
        BLOCK_W=0,  # will be overridden by autotune configs
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
        weight = self.weight.view(self.out_channels, self.kernel_size * self.kernel_size)
        return triton_depthwise_conv2d(x, weight, self.bias, self.stride, self.padding)
