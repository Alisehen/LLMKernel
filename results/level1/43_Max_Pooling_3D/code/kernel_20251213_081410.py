import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool3d_kernel(
    x_ptr,
    out_ptr,
    NC,
    ID,
    IH,
    IW,
    OD,
    OH,
    OW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    total_out,
    BLOCK: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < total_out

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    nc = tmp // OD

    d_start = od * stride_d - pad_d
    h_start = oh * stride_h - pad_h
    w_start = ow * stride_w - pad_w

    nc64 = nc.to(tl.int64)
    acc = tl.full([BLOCK], -float("inf"), tl.float32)

    for kd in range(KERNEL_D):
        cur_d = d_start + kd * dil_d
        valid_d = (cur_d >= 0) & (cur_d < ID)
        cur_d64 = cur_d.to(tl.int64)
        for kh in range(KERNEL_H):
            cur_h = h_start + kh * dil_h
            valid_h = (cur_h >= 0) & (cur_h < IH)
            cur_h64 = cur_h.to(tl.int64)
            for kw in range(KERNEL_W):
                cur_w = w_start + kw * dil_w
                valid_w = (cur_w >= 0) & (cur_w < IW)
                cur_w64 = cur_w.to(tl.int64)

                valid = mask & valid_d & valid_h & valid_w
                idx = ((nc64 * ID + cur_d64) * IH + cur_h64) * IW + cur_w64
                vals = tl.load(x_ptr + idx, mask=valid, other=-float("inf"))
                vals = vals.to(tl.float32)
                acc = tl.maximum(acc, vals)

    out_vals = acc.to(OUTPUT_DTYPE)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return tuple(int(v) for v in value)
    raise ValueError("Expected int or length-3 iterable.")


def _torch_dtype_to_triton(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise TypeError(f"Unsupported dtype: {dtype}")


def triton_maxpool3d(x, kernel_size, stride=None, padding=0, dilation=1):
    if not x.is_cuda:
        raise ValueError("Input tensor must be on CUDA device.")
    kernel = _triple(kernel_size)
    stride = _triple(kernel_size if stride is None else stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    N, C, ID, IH, IW = x.shape
    kD, kH, kW = kernel
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation

    def _out_dim(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    OD = _out_dim(ID, kD, sD, pD, dD)
    OH = _out_dim(IH, kH, sH, pH, dH)
    OW = _out_dim(IW, kW, sW, pW, dW)

    x_contig = x.contiguous()
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    total_out = out.numel()
    if total_out == 0:
        return out

    BLOCK = 128
    triton_dtype = _torch_dtype_to_triton(x.dtype)
    grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK"]),)

    maxpool3d_kernel[grid](
        x_contig,
        out,
        N * C,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        sD,
        sH,
        sW,
        pD,
        pH,
        pW,
        dD,
        dH,
        dW,
        total_out,
        BLOCK=BLOCK,
        KERNEL_D=kD,
        KERNEL_H=kH,
        KERNEL_W=kW,
        OUTPUT_DTYPE=triton_dtype,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool3d(x, self.kernel_size, self.stride, self.padding, self.dilation)
