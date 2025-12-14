# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_W": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_W": 192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_W": 256}, num_warps=8, num_stages=3),
    ],
    key=["NC", "OD", "OH", "OW"],
)
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
    BLOCK_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_plane = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)

    plane_per_nc = OD * OH
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < OW

    nc = pid_plane // plane_per_nc
    rem = pid_plane % plane_per_nc
    od = rem // OH
    oh = rem % OH

    d_start = od * stride_d - pad_d
    h_start = oh * stride_h - pad_h
    w_start = offs_w * stride_w - pad_w

    acc = tl.full([BLOCK_W], -float("inf"), tl.float32)

    stride_nc = ID * IH * IW
    stride_d_in = IH * IW
    stride_h_in = IW

    nc64 = nc.to(tl.int64)
    od64 = od.to(tl.int64)
    oh64 = oh.to(tl.int64)
    offs_w64 = offs_w.to(tl.int64)

    nc_base = nc64 * stride_nc

    for kd in range(KERNEL_D):
        cur_d = d_start + kd * dil_d
        valid_d = (cur_d >= 0) & (cur_d < ID)
        cur_d64 = cur_d.to(tl.int64)
        for kh in range(KERNEL_H):
            cur_h = h_start + kh * dil_h
            valid_h = (cur_h >= 0) & (cur_h < IH)
            cur_h64 = cur_h.to(tl.int64)
            base = nc_base + cur_d64 * stride_d_in + cur_h64 * stride_h_in
            base_mask = mask_w & valid_d & valid_h
            for kw in range(KERNEL_W):
                cur_w = w_start + kw * dil_w
                valid_w = (cur_w >= 0) & (cur_w < IW)
                total_mask = base_mask & valid_w
                cur_w64 = cur_w.to(tl.int64)
                vals = tl.load(x_ptr + base + cur_w64, mask=total_mask, other=-float("inf"))
                vals = vals.to(tl.float32)
                acc = tl.maximum(acc, vals)

    out_plane_offset = ((nc64 * OD + od64) * OH + oh64) * OW
    out_offsets = out_plane_offset + offs_w64
    out_vals = acc.to(OUTPUT_DTYPE)
    tl.store(out_ptr + out_offsets, out_vals, mask=mask_w)


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
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
    kD, kH, kW = _triple(kernel_size)
    sD, sH, sW = _triple(kernel_size if stride is None else stride)
    pD, pH, pW = _triple(padding)
    dD, dH, dW = _triple(dilation)

    N, C, ID, IH, IW = x.shape

    def _out_dim(in_size, k, s, p, d):
        return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

    OD = _out_dim(ID, kD, sD, pD, dD)
    OH = _out_dim(IH, kH, sH, pH, dH)
    OW = _out_dim(IW, kW, sW, pW, dW)

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        return out

    NC = N * C
    x_contig = x.contiguous()
    triton_dtype = _torch_dtype_to_triton(x.dtype)

    grid = lambda meta: (NC * OD * OH, triton.cdiv(OW, meta["BLOCK_W"]))

    maxpool3d_kernel[grid](
        x_contig,
        out,
        NC,
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
