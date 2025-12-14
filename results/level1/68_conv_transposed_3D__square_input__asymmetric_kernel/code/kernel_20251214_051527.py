# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Original configs
        triton.Config(
            {
                "BLOCK_OUT": 64,   # spatial / batch tile
                "BLOCK_COUT": 32,  # output-channel tile per group
            },
            num_warps=4,
            num_stages=2,
        ),
        # Added nearby config: same tile sizes, higher num_warps to
        # better utilize available warp slots on Ada (compute-bound kernel).
        triton.Config(
            {
                "BLOCK_OUT": 64,
                "BLOCK_COUT": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_OUT": 128,
                "BLOCK_COUT": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_OUT": 64,
                "BLOCK_COUT": 64,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_OUT": 128,
                "BLOCK_COUT": 64,
            },
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "D_OUT", "H_OUT", "W_OUT", "C_OUT_PER_GROUP"],
)
@triton.jit
def conv_transpose3d_kernel_optimized(
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
    BLOCK_COUT: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program IDs
    # axis 0 : tiles over N * D_out * H_out * W_out
    # axis 1 : tiles over C_out_per_group
    # axis 2 : over groups
    # -------------------------------------------------------------------------
    pid_spatial = tl.program_id(axis=0)
    pid_cout_blk = tl.program_id(axis=1)
    pid_group = tl.program_id(axis=2)

    group_id = pid_group

    # -------------------------------------------------------------------------
    # Spatial indices [BLOCK_OUT]
    # -------------------------------------------------------------------------
    total_spatial = N * D_OUT * H_OUT * W_OUT
    offs_spatial = pid_spatial * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_spatial = offs_spatial < total_spatial

    dhw = D_OUT * H_OUT * W_OUT
    hw = H_OUT * W_OUT

    n = offs_spatial // dhw
    tmp = offs_spatial % dhw
    od = tmp // hw
    tmp = tmp % hw
    oh = tmp // W_OUT
    ow = tmp % W_OUT

    # -------------------------------------------------------------------------
    # Output-channel indices within group [BLOCK_COUT]
    # -------------------------------------------------------------------------
    base_cout_in_group = pid_cout_blk * BLOCK_COUT
    offs_cout_in_group = base_cout_in_group + tl.arange(0, BLOCK_COUT)
    mask_cout = offs_cout_in_group < C_OUT_PER_GROUP

    # Global output-channel indices [BLOCK_COUT]
    c_out = group_id * C_OUT_PER_GROUP + offs_cout_in_group

    # -------------------------------------------------------------------------
    # Accumulator: [BLOCK_OUT, BLOCK_COUT] in fp32
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_OUT, BLOCK_COUT), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Optional bias: load once per (group, cout tile) and broadcast over spatial
    # -------------------------------------------------------------------------
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + c_out, mask=mask_cout, other=0.0)
        b_vals = b_vals.to(tl.float32)
        acc += b_vals[None, :]

    # -------------------------------------------------------------------------
    # Main loops over kernel and input channels
    # -------------------------------------------------------------------------
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

                valid_spatial = valid_d & valid_h & valid_w & mask_spatial

                # Precompute the part of the input offset that does not depend on c_in
                base_in_offset = (
                    n * x_stride_n
                    + id * x_stride_d
                    + ih * x_stride_h
                    + iw * x_stride_w
                )

                # Precompute kernel offset term independent of c_in and cout
                kernel_offset = (
                    kd * w_stride_kd
                    + kh * w_stride_kh
                    + kw * w_stride_kw
                )

                for c_in_g in range(C_IN_PER_GROUP):
                    c_in_global = group_id * C_IN_PER_GROUP + c_in_g

                    # Input: (n, c_in_global, id, ih, iw) for each spatial element
                    in_offset = base_in_offset + c_in_global * x_stride_c
                    x_vals = tl.load(
                        x_ptr + in_offset,
                        mask=valid_spatial,
                        other=0.0,
                    )
                    x_vals_f32 = x_vals.to(tl.float32)

                    # Weights: (c_in_global, cout_in_group, kd, kh, kw)
                    base_w_offset = c_in_global * w_stride_cin + kernel_offset
                    w_offset = base_w_offset + offs_cout_in_group * w_stride_cout
                    w_vals = tl.load(
                        w_ptr + w_offset,
                        mask=mask_cout,
                        other=0.0,
                    )
                    w_vals_f32 = w_vals.to(tl.float32)

                    # Outer product: [BLOCK_OUT] x [BLOCK_COUT] -> [BLOCK_OUT, BLOCK_COUT]
                    acc += x_vals_f32[:, None] * w_vals_f32[None, :]

    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------
    out_offset_spatial = (
        n * out_stride_n
        + od * out_stride_d
        + oh * out_stride_h
        + ow * out_stride_w
    )
    out_offset_c = c_out * out_stride_c
    out_offsets = out_offset_spatial[:, None] + out_offset_c[None, :]

    mask = mask_spatial[:, None] & mask_cout[None, :]

    tl.store(out_ptr + out_offsets, acc, mask=mask)


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

    # Strides
    x_strides = x.stride()
    w_strides = weight.stride()
    out_strides = out.stride()

    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty(1, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(N * D_out * H_out * W_out, meta["BLOCK_OUT"]),
            triton.cdiv(C_out_per_group, meta["BLOCK_COUT"]),
            groups,
        )

    conv_transpose3d_kernel_optimized[grid](
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
    Optimized for Ada (sm_89) GPUs with aggressive tiling over spatial and
    output-channel dimensions to maximize reuse and hide memory latency.
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
