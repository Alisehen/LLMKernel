import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=16, num_stages=2),
    ],
    key=["n_elements", "KD", "KH", "KW"],
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
    stride_n_elems,
    stride_c_elems,
    stride_d_elems,
    stride_h_elems,
    OUT_DTYPE: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    offsets = offsets.to(tl.int64)
    mask = offsets < n_elements

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    in_d_start = od * stride_d - pad_d
    in_h_start = oh * stride_h - pad_h
    in_w_start = ow * stride_w - pad_w

    base_nc = n * stride_n_elems + c * stride_c_elems
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    inv_window = 1.0 / (KD * KH * KW)

    for kd in tl.static_range(0, KD):
        cur_d = in_d_start + kd
        mask_d = mask & (cur_d >= 0) & (cur_d < D)
        d_base = base_nc + cur_d.to(tl.int64) * stride_d_elems

        for kh in tl.static_range(0, KH):
            cur_h = in_h_start + kh
            mask_dh = mask_d & (cur_h >= 0) & (cur_h < H)
            dh_base = d_base + cur_h.to(tl.int64) * stride_h_elems

            for kw in tl.static_range(0, KW):
                cur_w = in_w_start + kw
                mask_full = mask_dh & (cur_w >= 0) & (cur_w < W)
                ptr = dh_base + cur_w.to(tl.int64)
                vals = tl.load(x_ptr + ptr, mask=mask_full, other=0.0)
                sum_val += vals.to(tl.float32)

    avg = sum_val * inv_window

    if OUT_DTYPE == 0:
        avg_out = avg.to(tl.float16)
    elif OUT_DTYPE == 1:
        avg_out = avg
    elif OUT_DTYPE == 2:
        avg_out = avg.to(tl.bfloat16)
    else:
        avg_out = avg

    tl.store(y_ptr + offsets, avg_out, mask=mask)


def triton_avgpool3d(x, kernel_size, stride=None, padding=0):
    stride_for_torch = kernel_size if stride is None else stride
    if (not x.is_cuda) or x.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
        return torch.nn.functional.avg_pool3d(x, kernel_size, stride_for_torch, padding)

    kd, kh, kw = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    sd, sh, sw = (stride_for_torch, stride_for_torch, stride_for_torch) if isinstance(stride_for_torch, int) else tuple(stride_for_torch)
    pd, ph, pw = (padding, padding, padding) if isinstance(padding, int) else tuple(padding)

    x_contig = x.contiguous()
    N, C, D, H, W = x_contig.shape

    OD = (D + 2 * pd - kd) // sd + 1
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = out.numel()

    stride_c_elems = D * H * W
    stride_n_elems = stride_c_elems * C
    stride_d_elems = H * W
    stride_h_elems = W

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
        stride_n_elems,
        stride_c_elems,
        stride_d_elems,
        stride_h_elems,
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
