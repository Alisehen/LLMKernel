# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OUT": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 128}, num_warps=4, num_stages=2),
    ],
    key=["N", "D_OUT", "H_OUT", "W_OUT", "C_OUT_PER_GROUP"],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,        # (N, C_in, D_in, H_in, W_in)
    w_ptr,        # (C_in, C_out_per_group, Kd, Kh, Kw)
    b_ptr,        # (C_out,) or dummy
    out_ptr,      # (N, C_out, D_out, H_out, W_out)
    N,
    D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    GROUPS,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kd, w_stride_kh, w_stride_kw,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    C_IN_PER_GROUP: tl.constexpr,
    C_OUT_PER_GROUP: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # Program ids: grid = (N*D_out*H_out*W_out / BLOCK_OUT, C_out_per_group, groups)
    pid_spatial = tl.program_id(axis=0)        # over flattened (N, D_out, H_out, W_out)
    pid_cout_in_group = tl.program_id(axis=1)  # over C_out_per_group
    pid_group = tl.program_id(axis=2)          # over groups

    # Flat spatial indices for this program
    block_start = pid_spatial * BLOCK_OUT
    offs = block_start + tl.arange(0, BLOCK_OUT)
    total_spatial = N * D_OUT * H_OUT * W_OUT
    mask_o = offs < total_spatial

    # Decode n, od, oh, ow from flattened index
    tmp = offs
    dhw = D_OUT * H_OUT * W_OUT
    hw = H_OUT * W_OUT

    n = tmp // dhw
    tmp = tmp % dhw
    od = tmp // hw
    tmp = tmp % hw
    oh = tmp // W_OUT
    ow = tmp % W_OUT

    # Global output channel index
    group_id = pid_group
    c_out_in_group = pid_cout_in_group
    c_out = group_id * C_OUT_PER_GROUP + c_out_in_group

    # Per-lane base offsets for input/output
    in_n_offset = n * x_stride_n
    out_offset_base = (
        n * out_stride_n
        + c_out * out_stride_c
        + od * out_stride_d
        + oh * out_stride_h
        + ow * out_stride_w
    )

    # Accumulator (fp32)
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    # Optional bias (compile-time branch)
    if HAS_BIAS:
        b_val = tl.load(b_ptr + c_out)
        acc += b_val

    # Main computation: loop over kernel and input channels in this group
    for kd in range(KD):
        od_pad_kd = od + PAD_D - kd
        id = od_pad_kd // STRIDE_D
        valid_d = (od_pad_kd % STRIDE_D == 0) & (id >= 0) & (id < D_IN)

        for kh in range(KH):
            oh_pad_kh = oh + PAD_H - kh
            ih = oh_pad_kh // STRIDE_H
            valid_h = (oh_pad_kh % STRIDE_H == 0) & (ih >= 0) & (ih < H_IN)

            for kw in range(KW):
                ow_pad_kw = ow + PAD_W - kw
                iw = ow_pad_kw // STRIDE_W
                valid_w = (ow_pad_kw % STRIDE_W == 0) & (iw >= 0) & (iw < W_IN)

                # Combined spatial validity mask for this (kd, kh, kw)
                valid_spatial = valid_d & valid_h & valid_w & mask_o

                # Per-lane base offset for this (kd,kh,kw) position, without channel
                in_dhw_offset = (
                    in_n_offset
                    + id * x_stride_d
                    + ih * x_stride_h
                    + iw * x_stride_w
                )

                for c_in_g in range(C_IN_PER_GROUP):
                    c_in = group_id * C_IN_PER_GROUP + c_in_g

                    # Input index: (n, c_in, id, ih, iw)
                    in_offset = in_dhw_offset + c_in * x_stride_c
                    x_vals = tl.load(
                        x_ptr + in_offset,
                        mask=valid_spatial,
                        other=0.0,
                    )

                    # Weight index: (c_in, c_out_in_group, kd, kh, kw)
                    w_offset = (
                        c_in * w_stride_cin
                        + c_out_in_group * w_stride_cout
                        + kd * w_stride_kd
                        + kh * w_stride_kh
                        + kw * w_stride_kw
                    )
                    w_val = tl.load(w_ptr + w_offset).to(tl.float32)

                    acc += x_vals.to(tl.float32) * w_val

    # Store to output
    tl.store(out_ptr + out_offset_base, acc, mask=mask_o)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int,
) -> torch.Tensor:
    # Fallback for unsupported cases
    if (not x.is_cuda) or (x.dtype != torch.float32) or (weight.dtype != torch.float32):
        return torch.nn.functional.conv_transpose3d(
            x, weight, bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in, "Weight and input channel mismatch"
    assert C_in % groups == 0, "in_channels must be divisible by groups"
    C_out = C_out_per_group * groups
    C_in_per_group = C_in // groups

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    # Output size formula (PyTorch docs)
    D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w

    out = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    x_strides = x.stride()
    w_strides = weight.stride()
    out_strides = out.stride()

    has_bias = bias is not None
    if not has_bias:
        # Dummy tensor; will not be read when HAS_BIAS == 0
        bias = torch.empty(1, device=x.device, dtype=x.dtype)

    # Grid layout
    def grid(meta):
        return (
            triton.cdiv(N * D_out * H_out * W_out, meta["BLOCK_OUT"]),
            C_out_per_group,
            groups,
        )

    conv_transpose3d_kernel[grid](
        x,
        weight,
        bias,
        out,
        N,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        groups,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        w_strides[0], w_strides[1], w_strides[2], w_strides[3], w_strides[4],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
        C_IN_PER_GROUP=C_in_per_group,
        C_OUT_PER_GROUP=C_out_per_group,
        KD=Kd,
        KH=Kh,
        KW=Kw,
        HAS_BIAS=1 if has_bias else 0,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-based replacement for nn.ConvTranspose3d with asymmetric 3D kernels.
    Uses an optimized grid layout and autotuned spatial blocking.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
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
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        output_padding = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups

        return triton_conv_transpose3d(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )
