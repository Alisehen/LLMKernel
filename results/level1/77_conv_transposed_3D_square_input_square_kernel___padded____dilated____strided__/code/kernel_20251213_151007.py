import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
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
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M = N * OD * OH * OW

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < OC

    spatial = OD * OH * OW
    hw = OH * OW

    n_idx = m_offsets // spatial
    rem = m_offsets % spatial
    od_idx = rem // hw
    rem = rem % hw
    oh_idx = rem // OW
    ow_idx = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K_elems, BLOCK_K):
        k_offsets = k_base + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K_elems

        kw_idx = k_offsets % KW
        tmp = k_offsets // KW
        kh_idx = tmp % KH
        tmp = tmp // KH
        kd_idx = tmp % KD
        ic_idx = tmp // KD

        od_exp = od_idx[:, None]
        oh_exp = oh_idx[:, None]
        ow_exp = ow_idx[:, None]

        kd_exp = kd_idx[None, :]
        kh_exp = kh_idx[None, :]
        kw_exp = kw_idx[None, :]

        num_d = od_exp + pad_d - kd_exp * dil_d
        num_h = oh_exp + pad_h - kh_exp * dil_h
        num_w = ow_exp + pad_w - kw_exp * dil_w

        mask_d = (num_d >= 0) & (num_d % stride_d == 0)
        mask_h = (num_h >= 0) & (num_h % stride_h == 0)
        mask_w = (num_w >= 0) & (num_w % stride_w == 0)

        id_idx = num_d // stride_d
        ih_idx = num_h // stride_h
        iw_idx = num_w // stride_w

        mask_d &= (id_idx >= 0) & (id_idx < ID)
        mask_h &= (ih_idx >= 0) & (ih_idx < IH)
        mask_w &= (iw_idx >= 0) & (iw_idx < IW)

        valid_in = mask_m[:, None] & k_mask[None, :] & mask_d & mask_h & mask_w

        n_bc = n_idx[:, None].to(tl.int64)
        ic_bc = ic_idx[None, :].to(tl.int64)
        id_bc = id_idx.to(tl.int64)
        ih_bc = ih_idx.to(tl.int64)
        iw_bc = iw_idx.to(tl.int64)

        inp_offset = (((n_bc * IC + ic_bc) * ID + id_bc) * IH + ih_bc) * IW + iw_bc
        x_vals = tl.load(x_ptr + inp_offset, mask=valid_in, other=0.0)
        x_vals = x_vals.to(tl.float32)

        icw_bc = ic_idx[:, None].to(tl.int64)
        oc_bc = n_offsets[None, :].to(tl.int64)
        kd_bc = kd_idx[:, None].to(tl.int64)
        kh_bc = kh_idx[:, None].to(tl.int64)
        kw_bc = kw_idx[:, None].to(tl.int64)

        w_offset = (((icw_bc * OC + oc_bc) * KD + kd_bc) * KH + kh_bc) * KW + kw_bc
        w_vals = tl.load(w_ptr + w_offset, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        w_vals = w_vals.to(tl.float32)

        acc += tl.dot(x_vals, w_vals)

    out_mask = mask_m[:, None] & mask_n[None, :]

    n_bc = n_idx[:, None].to(tl.int64)
    oc_bc = n_offsets[None, :].to(tl.int64)
    od_bc = od_idx[:, None].to(tl.int64)
    oh_bc = oh_idx[:, None].to(tl.int64)
    ow_bc = ow_idx[:, None].to(tl.int64)

    out_offset = (((n_bc * OC + oc_bc) * OD + od_bc) * OH + oh_bc) * OW + ow_bc

    tl.store(out_ptr + out_offset, acc.to(OUT_DTYPE), mask=out_mask)


def _to_triton_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    raise TypeError(f"Unsupported dtype for Triton conv_transpose3d: {torch_dtype}")


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

    grid = (
        triton.cdiv(M, triton.next_power_of_2(min(128, M))),
        triton.cdiv(OC, triton.next_power_of_2(min(128, OC))),
    )

    out_dtype = _to_triton_dtype(x.dtype)

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
