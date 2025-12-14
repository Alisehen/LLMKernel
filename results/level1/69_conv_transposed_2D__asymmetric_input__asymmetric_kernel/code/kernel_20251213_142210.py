import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    groups,
    ic_per_group,
    oc_per_group,
    NUM_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_positions = N * OH * OW
    mask_m = offs_m < total_positions
    mask_n = offs_n < OC

    ohow = OH * OW
    n_idx = offs_m // ohow
    rem = offs_m % ohow
    oh_idx = rem // OW
    ow_idx = rem % OW

    oc_group = offs_n // oc_per_group
    oc_in_group = offs_n - oc_group * oc_per_group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(NUM_K):
        ic_base = k_iter * BLOCK_K
        ic_idx = ic_base + tl.arange(0, BLOCK_K)
        ic_mask = ic_idx < IC

        for kh in range(KH):
            dilated_kh = kh * dilation_h
            num_h = oh_idx + pad_h - dilated_kh
            ih = num_h // stride_h
            num_h_mod = num_h - ih * stride_h
            mask_h = (num_h >= 0) & (num_h_mod == 0) & (ih < H)

            for kw in range(KW):
                dilated_kw = kw * dilation_w
                num_w = ow_idx + pad_w - dilated_kw
                iw = num_w // stride_w
                num_w_mod = num_w - iw * stride_w
                mask_w = (num_w >= 0) & (num_w_mod == 0) & (iw < W)

                valid_pos = mask_m & mask_h & mask_w

                for kk in range(BLOCK_K):
                    ic_val = ic_idx[kk]
                    ic_valid = ic_mask[kk]
                    group_val = ic_val // ic_per_group
                    same_group = oc_group == group_val

                    pos_mask = valid_pos & ic_valid
                    weight_mask = mask_n & same_group & ic_valid

                    in_ptrs = (
                        x_ptr
                        + n_idx * in_stride_n
                        + ic_val * in_stride_c
                        + ih * in_stride_h
                        + iw * in_stride_w
                    )
                    inp_vals = tl.load(in_ptrs, mask=pos_mask, other=0.0)
                    w_ptrs = (
                        w_ptr
                        + ic_val * w_stride_ic
                        + oc_in_group * w_stride_oc
                        + kh * w_stride_h
                        + kw * w_stride_w
                    )
                    w_vals = tl.load(w_ptrs, mask=weight_mask, other=0.0)

                    acc += inp_vals[:, None].to(tl.float32) * w_vals[None, :].to(tl.float32)

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
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


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
    bias_ptr = bias if bias is not None else x

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

    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 8
    grid = (
        triton.cdiv(N * OH * OW, BLOCK_M),
        triton.cdiv(OC, BLOCK_N),
    )
    num_k = triton.cdiv(IC, BLOCK_K)

    conv_transpose2d_kernel[grid](
        x,
        weight,
        bias_ptr,
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
        groups,
        IC // groups,
        OC_per_group,
        NUM_K=num_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        KH=KH,
        KW=KW,
        HAS_BIAS=bias is not None,
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
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
