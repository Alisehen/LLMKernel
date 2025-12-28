import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,               # float32 [N, C_in, D_in, H_in, W_in]
    w_ptr,               # float32 [C_in, C_out_per_group, K_D, K_H, K_W]
    b_ptr,               # float32 [C_out] or unused
    y_ptr,               # float32 [N, C_out, D_out, H_out, W_out]
    N,                   # batch size
    D_IN, H_IN, W_IN,    # input spatial dims
    D_OUT, H_OUT, W_OUT, # output spatial dims
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    BLOCK_S: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    C_IN_PER_G: tl.constexpr,
    C_OUT_PER_G: tl.constexpr,
    C_IN_TOTAL: tl.constexpr,
    C_OUT_TOTAL: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(axis=0)  # over N * C_out_total
    pid_s = tl.program_id(axis=1)   # over spatial blocks

    # decode batch and output channel indices
    n = pid_nc // C_OUT_TOTAL
    oc = pid_nc % C_OUT_TOTAL

    # spatial indices (flattened)
    SPATIAL_OUT = D_OUT * H_OUT * W_OUT
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_out = offs_s < SPATIAL_OUT

    # decode (d, h, w) from flattened spatial index
    wh_out = H_OUT * W_OUT
    d_out_idx = offs_s // wh_out
    rem = offs_s % wh_out
    h_out_idx = rem // W_OUT
    w_out_idx = rem % W_OUT

    # initialize accumulator
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # determine group and per-group channel indices
    g = oc // C_OUT_PER_G
    oc_in_group = oc - g * C_OUT_PER_G
    ic_base = g * C_IN_PER_G

    # loop over input channels in this group
    for ci in tl.static_range(0, C_IN_PER_G):
        ic = ic_base + ci

        # base offset for this (n, ic, :, :, :)
        in_base = ((n * C_IN_TOTAL + ic) * D_IN * H_IN * W_IN)

        # base offset for weights for this (ic, oc_in_group, :, :, :)
        w_base = (ic * C_OUT_PER_G + oc_in_group) * (K_D * K_H * K_W)

        # loop over kernel depth
        for kd in tl.static_range(0, K_D):
            # compute input depth index mapping
            od_nom = d_out_idx + PAD_D - kd
            valid_d = (od_nom >= 0) & (od_nom < D_IN * STRIDE_D)
            # ensure stride alignment
            valid_d = valid_d & ((od_nom % STRIDE_D) == 0)
            od = od_nom // STRIDE_D

            # loop over kernel height
            for kh in tl.static_range(0, K_H):
                oh_nom = h_out_idx + PAD_H - kh
                valid_h = (oh_nom >= 0) & (oh_nom < H_IN * STRIDE_H)
                valid_h = valid_h & ((oh_nom % STRIDE_H) == 0)
                oh = oh_nom // STRIDE_H

                # loop over kernel width
                for kw in tl.static_range(0, K_W):
                    ow_nom = w_out_idx + PAD_W - kw
                    valid_w = (ow_nom >= 0) & (ow_nom < W_IN * STRIDE_W)
                    valid_w = valid_w & ((ow_nom % STRIDE_W) == 0)
                    ow = ow_nom // STRIDE_W

                    mask = mask_out & valid_d & valid_h & valid_w

                    # linear input offsets for this kernel position
                    in_offsets = in_base + ((od * H_IN + oh) * W_IN + ow)

                    x_vals = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)

                    w_idx = w_base + kd * (K_H * K_W) + kh * K_W + kw
                    w_val = tl.load(w_ptr + w_idx)

                    acc += x_vals * w_val

    if HAS_BIAS:
        # add bias for this output channel
        b_val = tl.load(b_ptr + oc)
        acc = acc + b_val

    # store results
    out_base = ((n * C_OUT_TOTAL + oc) * D_OUT * H_OUT * W_OUT)
    out_offsets = out_base + offs_s
    tl.store(y_ptr + out_offsets, acc, mask=mask_out)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    groups: int,
) -> torch.Tensor:
    """
    x:       (N, C_in, D_in, H_in, W_in)
    weight:  (C_in, C_out_per_group, K_D, K_H, K_W)
    bias:    (C_out,) or None
    stride:  (s_d, s_h, s_w)
    padding: (p_d, p_h, p_w)
    groups:  int
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv_transpose3d requires CUDA tensors"

    # ensure contiguous and float32 for compute
    device = x.device
    x_fp32 = x.contiguous().to(torch.float32)
    w_fp32 = weight.contiguous().to(torch.float32)
    b_fp32 = None
    has_bias = bias is not None
    if has_bias:
        b_fp32 = bias.contiguous().to(torch.float32)

    N, C_in, D_in, H_in, W_in = x_fp32.shape
    C_in_w, C_out_per_g, K_D, K_H, K_W = w_fp32.shape
    assert C_in_w == C_in, "weight C_in mismatch"

    s_d, s_h, s_w = stride
    p_d, p_h, p_w = padding

    C_out = C_out_per_g * groups

    # output dims: (Lin - 1) * s - 2*p + k
    D_out = (D_in - 1) * s_d - 2 * p_d + K_D
    H_out = (H_in - 1) * s_h - 2 * p_h + K_H
    W_out = (W_in - 1) * s_w - 2 * p_w + K_W

    y_fp32 = torch.empty((N, C_out, D_out, H_out, W_out),
                         device=device, dtype=torch.float32)

    C_in_per_g = C_in // groups
    C_out_per_g = C_out // groups
    C_in_total = C_in
    C_out_total = C_out

    SPATIAL_OUT = D_out * H_out * W_out
    BLOCK_S = 128

    grid = (
        N * C_out_total,
        triton.cdiv(SPATIAL_OUT, BLOCK_S),
    )

    conv_transpose3d_kernel[grid](
        x_fp32, w_fp32, b_fp32 if has_bias else x_fp32, y_fp32,
        N,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        s_d, s_h, s_w,
        p_d, p_h, p_w,
        BLOCK_S=BLOCK_S,
        K_D=K_D,
        K_H=K_H,
        K_W=K_W,
        C_IN_PER_G=C_in_per_g,
        C_OUT_PER_G=C_out_per_g,
        C_IN_TOTAL=C_in_total,
        C_OUT_TOTAL=C_out_total,
        GROUPS=groups,
        HAS_BIAS=has_bias,
    )

    # cast back to original dtype if needed
    if x.dtype != torch.float32:
        return y_fp32.to(x.dtype)
    return y_fp32


class ModelNew(nn.Module):
    """
    ConvTranspose3d implemented with a high-performance Triton kernel.
    Matches the API and initialization of the original PyTorch module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Use PyTorch ConvTranspose3d module only to manage parameters
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract parameters
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        groups = self.conv_transpose3d.groups

        return triton_conv_transpose3d(x, w, b, stride, padding, groups)
