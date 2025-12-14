import torch
import torch.nn as nn
import triton
import triton.language as tl


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise NotImplementedError(f"Unsupported dtype {dtype}")


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    L_IN,
    L_OUT,
    stride,
    padding,
    dilation,
    x_stride_batch,
    x_stride_ch,
    x_stride_l,
    w_stride_in,
    w_stride_out,
    w_stride_k,
    out_stride_batch,
    out_stride_ch,
    out_stride_l,
    inv_stride,
    has_bias: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    batch = pid0 // C_OUT
    oc = pid0 % C_OUT

    if batch >= B:
        return

    t_offsets = pid1 * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_OUT
    t_offsets_i32 = t_offsets.to(tl.int32)

    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    if has_bias:
        bias_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += bias_val

    tmp = t_offsets_i32 + padding
    stride_val = stride
    dilation_val = dilation

    batch_offset = batch.to(tl.int64) * x_stride_batch
    out_base = out_ptr + batch.to(tl.int64) * out_stride_batch + oc.to(tl.int64) * out_stride_ch
    tmp_out_ptrs = out_base + t_offsets.to(tl.int64) * out_stride_l

    for cin in range(C_IN):
        x_base = batch_offset + cin * x_stride_ch
        cin_weight_offset = cin * w_stride_in + oc * w_stride_out
        for k in range(K):
            numerator = tmp - k * dilation_val
            numerator_i32 = numerator
            valid = mask_t & (numerator_i32 >= 0)
            numerator_f = numerator_i32.to(tl.float32)
            l_float = numerator_f * inv_stride
            l_floor = tl.floor(l_float)
            l_int = l_floor.to(tl.int32)
            valid = valid & (numerator_i32 == l_int * stride_val) & (l_int >= 0) & (l_int < L_IN)
            l_offsets = l_int.to(tl.int64) * x_stride_l
            x_vals = tl.load(x_ptr + x_base + l_offsets, mask=valid, other=0.0)
            w_val = tl.load(w_ptr + cin_weight_offset + k * w_stride_k)
            acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    tl.store(tmp_out_ptrs, acc.to(OUT_DTYPE), mask=mask_t)


def conv1d_transpose_triton(x, weight, bias, stride, padding, dilation):
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_IN, L_IN = x.shape
    _, C_OUT, K = weight.shape

    L_OUT = (L_IN - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    out = torch.empty((B, C_OUT, L_OUT), device=x.device, dtype=x.dtype)

    BLOCK_T = 128
    grid = (B * C_OUT, triton.cdiv(L_OUT, BLOCK_T))

    x_stride_batch, x_stride_ch, x_stride_l = x.stride()
    w_stride_in, w_stride_out, w_stride_k = weight.stride()
    out_stride_batch, out_stride_ch, out_stride_l = out.stride()

    bias_ptr = bias if bias is not None else out
    has_bias = bias is not None

    conv1d_transpose_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        B,
        L_IN,
        L_OUT,
        stride,
        padding,
        dilation,
        x_stride_batch,
        x_stride_ch,
        x_stride_l,
        w_stride_in,
        w_stride_out,
        w_stride_k,
        out_stride_batch,
        out_stride_ch,
        out_stride_l,
        1.0 / float(stride),
        has_bias=has_bias,
        C_IN=C_IN,
        C_OUT=C_OUT,
        K=K,
        BLOCK_T=BLOCK_T,
        OUT_DTYPE=_torch_dtype_to_triton(out.dtype),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv1d_transpose_triton(
            x,
            self.conv1d_transpose.weight,
            self.conv1d_transpose.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
