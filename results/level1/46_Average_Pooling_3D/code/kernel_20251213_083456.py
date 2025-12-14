import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
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

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    count = tl.zeros([BLOCK], dtype=tl.float32)

    for kd in range(KD):
        cur_d = in_d_start + kd
        valid_d = (cur_d >= 0) & (cur_d < D)
        d_offset = cur_d.to(tl.int64) * stride_d_in
        for kh in range(KH):
            cur_h = in_h_start + kh
            valid_dh = valid_d & (cur_h >= 0) & (cur_h < H)
            h_offset = cur_h.to(tl.int64) * stride_h_in
            for kw in range(KW):
                cur_w = in_w_start + kw
                valid = mask & valid_dh & (cur_w >= 0) & (cur_w < W)
                w_offset = cur_w.to(tl.int64)
                input_offset = base_nc + d_offset + h_offset + w_offset
                vals = tl.load(x_ptr + input_offset, mask=valid, other=0.0)
                sum_val += vals.to(tl.float32)
                count += valid.to(tl.float32)

    nonzero = count > 0
    inv_count = tl.where(nonzero, 1.0 / count, 0.0)
    avg = sum_val * inv_count

    if OUT_DTYPE == 0:
        avg_out = avg.to(tl.float16)
    elif OUT_DTYPE == 1:
        avg_out = avg
    elif OUT_DTYPE == 2:
        avg_out = avg.to(tl.bfloat16)
    else:
        avg_out = avg

    tl.store(y_ptr + offsets, avg_out, mask=mask)


def _as_triple(value, name):
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return tuple(int(v) for v in value)
    raise ValueError(f"{name} must be an int or a sequence of length 3, got {value}.")


def triton_avgpool3d(x, kernel_size, stride=None, padding=0):
    stride_for_torch = kernel_size if stride is None else stride
    if (not x.is_cuda) or x.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        return F.avg_pool3d(x, kernel_size, stride_for_torch, padding)

    kd, kh, kw = _as_triple(kernel_size, "kernel_size")
    sd, sh, sw = _as_triple(stride_for_torch, "stride")
    pd, ph, pw = _as_triple(padding, "padding")

    assert x.dim() == 5, "Input tensor must be 5-dimensional (N, C, D, H, W)."
    x_contig = x.contiguous()
    N, C, D, H, W = x_contig.shape

    OD = (D + 2 * pd - kd) // sd + 1
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1

    out = torch.empty((N, C, OD, OH, OW), device=x_contig.device, dtype=x_contig.dtype)
    n_elements = out.numel()

    dtype_map = {
        torch.float16: 0,
        torch.float32: 1,
        torch.bfloat16: 2,
    }
    out_dtype = dtype_map[x_contig.dtype]

    BLOCK = 128

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
        BLOCK=BLOCK,
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
