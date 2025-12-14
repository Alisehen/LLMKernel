import torch
import torch.nn as nn
import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2, num_ctas=2),
]


@triton.autotune(configs=autotune_configs, key=["total_in_elems"])
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    total_in_elems,
    N,
    Cin,
    Cout,
    D,
    H,
    W,
    KD,
    KH,
    KW,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    out_pad_d,
    out_pad_h,
    out_pad_w,
    out_D,
    out_H,
    out_W,
    cin_per_group,
    cout_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_in_elems
    offsets64 = offsets.to(tl.int64)

    w_idx = offsets64 % W
    tmp = offsets64 // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    tmp = tmp // D
    c_idx = tmp % Cin
    n_idx = tmp // Cin

    x_vals = tl.load(x_ptr + offsets64, mask=mask, other=0.0)

    group_idx = c_idx // cin_per_group
    oc_base = group_idx * cout_per_group

    for kd in range(0, KD):
        out_d = d_idx * stride_d - pad_d + kd
        valid_d = (out_d >= 0) & (out_d < out_D)
        for kh in range(0, KH):
            out_h = h_idx * stride_h - pad_h + kh
            valid_h = (out_h >= 0) & (out_h < out_H)
            for kw in range(0, KW):
                out_w = w_idx * stride_w - pad_w + kw
                valid_w = (out_w >= 0) & (out_w < out_W)
                base_mask = mask & valid_d & valid_h & valid_w

                for oc_sub in range(0, cout_per_group):
                    oc = oc_base + oc_sub
                    w_offset = (((((c_idx * cout_per_group) + oc_sub) * KD + kd) * KH + kh) * KW + kw)
                    w_vals = tl.load(w_ptr + w_offset, mask=base_mask, other=0.0)

                    out_offset = (((((n_idx * Cout) + oc) * out_D + out_d) * out_H + out_h) * out_W + out_w)
                    contrib = x_vals * w_vals
                    tl.atomic_add(out_ptr + out_offset, contrib, mask=base_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    N, Cin, D, H, W = x.shape
    KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding
    Cout = weight.shape[1] * groups

    out_D = (D - 1) * stride_d - 2 * pad_d + KD + out_pad_d
    out_H = (H - 1) * stride_h - 2 * pad_h + KH + out_pad_h
    out_W = (W - 1) * stride_w - 2 * pad_w + KW + out_pad_w

    output = torch.zeros((N, Cout, out_D, out_H, out_W), device=x.device, dtype=x.dtype)

    total_in_elems = x.numel()
    BLOCK_SIZE = 128

    def grid(meta):
        return (triton.cdiv(total_in_elems, meta["BLOCK_SIZE"]),)

    conv_transpose3d_kernel[grid](
        x,
        weight,
        output,
        total_in_elems,
        N,
        Cin,
        Cout,
        D,
        H,
        W,
        KD,
        KH,
        KW,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        out_pad_d,
        out_pad_h,
        out_pad_w,
        out_D,
        out_H,
        out_W,
        Cin // groups,
        Cout // groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if bias is not None:
        output += bias.view(1, Cout, 1, 1, 1)

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1),
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.conv_transpose3d.weight,
            self.conv_transpose3d.bias,
            self.conv_transpose3d.stride,
            self.conv_transpose3d.padding,
            self.conv_transpose3d.output_padding,
            self.conv_transpose3d.groups,
        )
