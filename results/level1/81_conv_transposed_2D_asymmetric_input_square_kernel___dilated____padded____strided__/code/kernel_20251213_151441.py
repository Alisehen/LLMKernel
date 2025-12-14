# import section (fixed order)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 96},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 96},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 96, "BLOCK_N": 192, "BLOCK_K": 64},
            num_warps=6,
            num_stages=3,
        ),
    ],
    key=["N", "OC", "OH", "OW"],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    N,
    IC,
    IH,
    IW,
    OC,
    KH,
    KW,
    OH,
    OW,
    STRIDE_H,
    STRIDE_W,
    PAD_H,
    PAD_W,
    DIL_H,
    DIL_W,
    K,
    ADD_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    ohow = OH * OW
    tot_m = N * ohow

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < tot_m
    n_mask = n_offsets < OC

    n_idx = tl.where(m_mask, m_offsets // ohow, 0)
    rem = m_offsets - n_idx * ohow
    oh_idx = rem // OW
    ow_idx = rem - oh_idx * OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        ic_kernel = KH * KW
        ic_idx = k_offsets // ic_kernel
        kwkh = k_offsets - ic_idx * ic_kernel
        kh_idx = kwkh // KW
        kw_idx = kwkh - kh_idx * KW

        oh_pad = oh_idx[:, None] + PAD_H
        ow_pad = ow_idx[:, None] + PAD_W

        ih_nom = oh_pad - kh_idx[None, :] * DIL_H
        iw_nom = ow_pad - kw_idx[None, :] * DIL_W

        ih_div = tl.math.floor_div(ih_nom, STRIDE_H)
        iw_div = tl.math.floor_div(iw_nom, STRIDE_W)

        valid_h = (ih_nom >= 0) & (ih_nom == ih_div * STRIDE_H) & (ih_div < IH)
        valid_w = (iw_nom >= 0) & (iw_nom == iw_div * STRIDE_W) & (iw_div < IW)
        valid = m_mask[:, None] & k_mask[None, :] & valid_h & valid_w

        n_i64 = n_idx.to(tl.int64)[:, None]
        ic_i64 = ic_idx.to(tl.int64)[None, :]
        ih_i64 = ih_div.to(tl.int64)
        iw_i64 = iw_div.to(tl.int64)

        x_ptrs = (((n_i64 * IC) + ic_i64) * IH + ih_i64) * IW + iw_i64
        x_vals = tl.load(x_ptr + x_ptrs, mask=valid, other=0.0).to(tl.float16)

        w_mask = k_mask[:, None] & n_mask[None, :]
        n_i64_w = n_offsets.to(tl.int64)[None, :]
        kh_i64 = kh_idx.to(tl.int64)[:, None]
        kw_i64 = kw_idx.to(tl.int64)[:, None]

        w_ptrs = (
            (((ic_idx.to(tl.int64)[:, None] * OC) + n_i64_w) * KH + kh_i64) * KW + kw_i64
        )
        w_vals = tl.load(w_ptr + w_ptrs, mask=w_mask, other=0.0).to(tl.float16)

        acc += tl.dot(x_vals, w_vals, out_dtype=tl.float32)

    if ADD_BIAS:
        bias_vals = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    out_mask = m_mask[:, None] & n_mask[None, :]
    n_idx64 = n_idx.to(tl.int64)[:, None]
    oh_i64 = oh_idx.to(tl.int64)[:, None]
    ow_i64 = ow_idx.to(tl.int64)[:, None]
    n_offsets_i64 = n_offsets.to(tl.int64)[None, :]

    y_ptrs = (((n_idx64 * OC) + n_offsets_i64) * OH + oh_i64) * OW + ow_i64
    tl.store(y_ptr + y_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv_transpose2d(x, weight, bias, stride, padding, dilation):
    stride_h, stride_w = (stride if isinstance(stride, tuple) else (stride, stride))
    pad_h, pad_w = (padding if isinstance(padding, tuple) else (padding, padding))
    dil_h, dil_w = (dilation if isinstance(dilation, tuple) else (dilation, dilation))

    x = x.contiguous()
    weight = weight.contiguous()
    bias_tensor = bias.contiguous() if bias is not None else None

    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC_w == IC

    OH = (IH - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1
    OW = (IW - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    K = IC * KH * KW
    bias_ptr = bias_tensor if bias_tensor is not None else out.new_empty(1)

    def grid(meta):
        return (
            triton.cdiv(N * OH * OW, meta["BLOCK_M"]),
            triton.cdiv(OC, meta["BLOCK_N"]),
        )

    conv_transpose2d_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        N,
        IC,
        IH,
        IW,
        OC,
        KH,
        KW,
        OH,
        OW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        K,
        ADD_BIAS=1 if bias_tensor is not None else 0,
    )
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
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.conv.weight,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
        )
