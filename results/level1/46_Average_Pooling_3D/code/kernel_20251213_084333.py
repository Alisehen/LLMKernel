# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def avg_pool3d_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    OUT_DTYPE: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    stride_c = D * H * W
    stride_n = stride_c * C
    stride_d_in = H * W
    stride_h_in = W

    in_d_start = od * stride_d - pad_d
    in_h_start = oh * stride_h - pad_h
    in_w_start = ow * stride_w - pad_w

    base_nc = n.to(tl.int64) * stride_n + c.to(tl.int64) * stride_c

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    inv_window = 1.0 / (KD * KH * KW)

    for kd in tl.static_range(0, KD):
        cur_d = in_d_start + kd
        valid_d = (cur_d >= 0) & (cur_d < D)
        d_offset = tl.where(valid_d, cur_d, 0).to(tl.int64) * stride_d_in

        for kh in tl.static_range(0, KH):
            cur_h = in_h_start + kh
            valid_dh = valid_d & (cur_h >= 0) & (cur_h < H)
            h_offset = tl.where(valid_dh, cur_h, 0).to(tl.int64) * stride_h_in

            for kw in tl.static_range(0, KW):
                cur_w = in_w_start + kw
                valid = mask & valid_dh & (cur_w >= 0) & (cur_w < W)
                w_offset = tl.where(valid, cur_w, 0).to(tl.int64)

                input_offset = base_nc + d_offset + h_offset + w_offset
                vals = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
                acc += vals.to(tl.float32)

    avg = acc * inv_window

    if OUT_DTYPE == 0:
        avg = avg.to(tl.float16)
    elif OUT_DTYPE == 2:
        avg = avg.to(tl.bfloat16)

    tl.store(y_ptr + offsets, avg, mask=mask)


def triton_avgpool3d(x, kernel_size, stride=None, padding=0):
    stride = kernel_size if stride is None else stride

    def _triple(val):
        if isinstance(val, int):
            return (val, val, val)
        if len(val) == 3:
            return tuple(val)
        raise ValueError("Expected int or length-3 tuple")

    kd, kh, kw = _triple(kernel_size)
    sd, sh, sw = _triple(stride)
    pd, ph, pw = _triple(padding)

    if (not x.is_cuda) or x.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        return torch.nn.functional.avg_pool3d(x, (kd, kh, kw), (sd, sh, sw), (pd, ph, pw))

    x_contig = x.contiguous()
    N, C, D, H, W = x_contig.shape

    OD = (D + 2 * pd - kd) // sd + 1
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = out.numel()

    dtype_map = {torch.float16: 0, torch.float32: 1, torch.bfloat16: 2}
    out_dtype = dtype_map[out.dtype]

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)

    avg_pool3d_kernel[grid](
        x_contig,
        out,
        n_elements,
        N,
        C,
        D,
        H,
        W,
        OD,
        OH,
        OW,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        OUT_DTYPE=out_dtype,
        KD=kd,
        KH=kh,
        KW=kw,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avgpool3d(x, self.kernel_size, self.stride, self.padding)
