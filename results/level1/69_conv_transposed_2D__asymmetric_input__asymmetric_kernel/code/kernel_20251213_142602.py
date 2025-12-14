# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["N", "IC", "OC", "OH", "OW"],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    in_stride_n,
    in_stride_c,
    in_stride_h,
    in_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    w_stride_ic,
    w_stride_oc,
    w_stride_h,
    w_stride_w,
    N,
    IC,
    OC,
    H,
    W,
    OH,
    OW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    ic_per_group,
    oc_per_group,
    K,
    HAS_BIAS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_positions = N * OH * OW
    mask_m = offs_m < total_positions
    mask_n = offs_n < OC

    ohow = OH * OW
    n_idx = tl.where(mask_m, offs_m // ohow, 0)
    rem = offs_m - n_idx * ohow
    oh_idx = tl.where(mask_m, rem // OW, 0)
    ow_idx = tl.where(mask_m, rem - oh_idx * OW, 0)

    oc_group = tl.where(mask_n, offs_n // oc_per_group, 0)
    oc_in_group = tl.where(mask_n, offs_n - oc_group * oc_per_group, 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K, BLOCK_K):
        k_offsets = k_base + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        k_offsets = tl.where(k_mask, k_offsets, 0)

        kw_idx = k_offsets % KW
        tmp = k_offsets // KW
        kh_idx = tmp % KH
        ic_idx = tmp // KH

        kw_idx = tl.where(k_mask, kw_idx, 0)
        kh_idx = tl.where(k_mask, kh_idx, 0)
        ic_idx = tl.where(k_mask, ic_idx, 0)
        ic_group = tl.where(k_mask, ic_idx // ic_per_group, 0)

        oh = oh_idx[:, None]
        ow = ow_idx[:, None]

        num_h = oh + pad_h - kh_idx[None, :] * dilation_h
        mask_h = num_h >= 0
        safe_num_h = tl.where(mask_h, num_h, 0)
        ih = safe_num_h // stride_h
        mask_h = mask_h & (safe_num_h - ih * stride_h == 0) & (ih < H)

        num_w = ow + pad_w - kw_idx[None, :] * dilation_w
        mask_w = num_w >= 0
        safe_num_w = tl.where(mask_w, num_w, 0)
        iw = safe_num_w // stride_w
        mask_w = mask_w & (safe_num_w - iw * stride_w == 0) & (iw < W)

        valid_mk = mask_m[:, None] & k_mask[None, :] & mask_h & mask_w
        same_group = ic_group[:, None] == oc_group[None, :]
        weight_mask = k_mask[:, None] & mask_n[None, :] & same_group

        in_ptrs = (
            x_ptr
            + n_idx[:, None] * in_stride_n
            + ic_idx[None, :] * in_stride_c
            + ih * in_stride_h
            + iw * in_stride_w
        )
        inp_tile = tl.load(in_ptrs, mask=valid_mk, other=0.0)

        w_ptrs = (
            w_ptr
            + ic_idx[:, None] * w_stride_ic
            + oc_in_group[None, :] * w_stride_oc
            + kh_idx[:, None] * w_stride_h
            + kw_idx[:, None] * w_stride_w
        )
        w_tile = tl.load(w_ptrs, mask=weight_mask, other=0.0)

        acc += tl.dot(inp_tile, w_tile)

    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    out_ptrs = (
        y_ptr
        + n_idx[:, None] * out_stride_n
        + offs_n[None, :] * out_stride_c
        + oh_idx[:, None] * out_stride_h
        + ow_idx[:, None] * out_stride_w
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(OUT_DTYPE), mask=out_mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1),
    padding=(0, 0),
    output_padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
):
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, IC, H, W = x.shape
    _, OC_per_group, KH, KW = weight.shape
    OC = OC_per_group * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    op_h, op_w = output_padding
    dilation_h, dilation_w = dilation

    OH = (H - 1) * stride_h - 2 * pad_h + dilation_h * (KH - 1) + op_h + 1
    OW = (W - 1) * stride_w - 2 * pad_w + dilation_w * (KW - 1) + op_w + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()
    w_stride_ic, w_stride_oc, w_stride_h, w_stride_w = weight.stride()

    total_k = IC * KH * KW
    ic_per_group = IC // groups

    grid = lambda META: (
        triton.cdiv(N * OH * OW, META["BLOCK_M"]),
        triton.cdiv(OC, META["BLOCK_N"]),
    )

    conv_transpose2d_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out,
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        w_stride_ic,
        w_stride_oc,
        w_stride_h,
        w_stride_w,
        N,
        IC,
        OC,
        H,
        W,
        OH,
        OW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ic_per_group,
        OC_per_group,
        total_k,
        HAS_BIAS=bias is not None,
        OUT_DTYPE=_torch_dtype_to_triton(out.dtype),
        KH=KH,
        KW=KW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels // groups,
                kernel_size[0],
                kernel_size[1],
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        if bias:
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
