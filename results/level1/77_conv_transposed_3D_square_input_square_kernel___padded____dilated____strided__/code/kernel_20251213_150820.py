# import section (must be first)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["N", "OC", "OD", "OH", "OW", "IC"],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    IC,
    ID,
    IH,
    IW,
    OC,
    KD,
    KH,
    KW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    OD,
    OH,
    OW,
    K_elems,
    x_batch_stride,
    x_chan_stride,
    x_d_stride,
    x_h_stride,
    w_chan_stride,
    w_out_stride,
    w_d_stride,
    w_h_stride,
    w_w_stride,
    out_batch_stride,
    out_chan_stride,
    out_d_stride,
    out_h_stride,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M = N * OD * OH * OW
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_idx < M
    n_mask = n_idx_out < OC

    spatial = OD * OH * OW
    hw = OH * OW

    n_batch = m_idx // spatial
    rem = m_idx % spatial
    od = rem // hw
    rem = rem % hw
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K_elems, BLOCK_K):
        k_idx = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K_elems

        kw = k_idx % KW
        tmp = k_idx // KW
        kh = tmp % KH
        tmp = tmp // KH
        kd = tmp % KD
        ic = tmp // KD

        od_exp = od[:, None]
        oh_exp = oh[:, None]
        ow_exp = ow[:, None]

        kd_exp = kd[None, :]
        kh_exp = kh[None, :]
        kw_exp = kw[None, :]

        num_d = od_exp + pad_d - kd_exp * dil_d
        num_h = oh_exp + pad_h - kh_exp * dil_h
        num_w = ow_exp + pad_w - kw_exp * dil_w

        can_div_d = (num_d >= 0) & ((num_d % stride_d) == 0)
        can_div_h = (num_h >= 0) & ((num_h % stride_h) == 0)
        can_div_w = (num_w >= 0) & ((num_w % stride_w) == 0)

        id_idx = num_d // stride_d
        ih_idx = num_h // stride_h
        iw_idx = num_w // stride_w

        valid_d = can_div_d & (id_idx >= 0) & (id_idx < ID)
        valid_h = can_div_h & (ih_idx >= 0) & (ih_idx < IH)
        valid_w = can_div_w & (iw_idx >= 0) & (iw_idx < IW)

        valid_in = m_mask[:, None] & k_mask[None, :] & valid_d & valid_h & valid_w

        nb = n_batch[:, None].to(tl.int64)
        ic_bc = ic[None, :].to(tl.int64)
        id_bc = id_idx.to(tl.int64)
        ih_bc = ih_idx.to(tl.int64)
        iw_bc = iw_idx.to(tl.int64)

        x_offset = (
            nb * x_batch_stride
            + ic_bc * x_chan_stride
            + id_bc * x_d_stride
            + ih_bc * x_h_stride
            + iw_bc
        )
        x_vals = tl.load(x_ptr + x_offset, mask=valid_in, other=0.0).to(tl.float32)

        ic_w = ic[:, None].to(tl.int64)
        oc_bc = n_idx_out[None, :].to(tl.int64)
        kd_bc = kd[:, None].to(tl.int64)
        kh_bc = kh[:, None].to(tl.int64)
        kw_bc = kw[:, None].to(tl.int64)

        w_offset = (
            ic_w * w_chan_stride
            + oc_bc * w_out_stride
            + kd_bc * w_d_stride
            + kh_bc * w_h_stride
            + kw_bc * w_w_stride
        )
        w_vals = tl.load(w_ptr + w_offset, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(x_vals, w_vals)

    out_mask = m_mask[:, None] & n_mask[None, :]

    nb = n_batch[:, None].to(tl.int64)
    oc_bc = n_idx_out[None, :].to(tl.int64)
    od_bc = od[:, None].to(tl.int64)
    oh_bc = oh[:, None].to(tl.int64)
    ow_bc = ow[:, None].to(tl.int64)

    out_offset = (
        nb * out_batch_stride
        + oc_bc * out_chan_stride
        + od_bc * out_d_stride
        + oh_bc * out_h_stride
        + ow_bc
    )
    tl.store(out_ptr + out_offset, acc.to(OUT_DTYPE), mask=out_mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, dilation):
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors."
    assert x.dtype == weight.dtype, "Input and weight dtypes must match."

    stride_d, stride_h, stride_w = (stride if isinstance(stride, tuple) else (stride,) * 3)
    pad_d, pad_h, pad_w = (padding if isinstance(padding, tuple) else (padding,) * 3)
    dil_d, dil_h, dil_w = (dilation if isinstance(dilation, tuple) else (dilation,) * 3)

    x = x.contiguous()
    weight = weight.contiguous()

    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape

    OD = (ID - 1) * stride_d - 2 * pad_d + dil_d * (KD - 1) + 1
    OH = (IH - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1
    OW = (IW - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OD * OH * OW
    K_elems = IC * KD * KH * KW

    x_batch_stride = IC * ID * IH * IW
    x_chan_stride = ID * IH * IW
    x_d_stride = IH * IW
    x_h_stride = IW

    w_chan_stride = OC * KD * KH * KW
    w_out_stride = KD * KH * KW
    w_d_stride = KH * KW
    w_h_stride = KW
    w_w_stride = 1

    out_batch_stride = OC * OD * OH * OW
    out_chan_stride = OD * OH * OW
    out_d_stride = OH * OW
    out_h_stride = OW

    out_dtype = tl.float32 if x.dtype == torch.float32 else tl.float16 if x.dtype == torch.float16 else tl.bfloat16

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(OC, meta["BLOCK_N"]),
    )

    conv_transpose3d_kernel[grid](
        x,
        weight,
        out,
        N,
        IC,
        ID,
        IH,
        IW,
        OC,
        KD,
        KH,
        KW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        OD,
        OH,
        OW,
        K_elems,
        x_batch_stride,
        x_chan_stride,
        x_d_stride,
        x_h_stride,
        w_chan_stride,
        w_out_stride,
        w_d_stride,
        w_h_stride,
        w_w_stride,
        out_batch_stride,
        out_chan_stride,
        out_d_stride,
        out_h_stride,
        OUT_DTYPE=out_dtype,
    )

    if bias is not None:
        out += bias.view(1, -1, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
        )
