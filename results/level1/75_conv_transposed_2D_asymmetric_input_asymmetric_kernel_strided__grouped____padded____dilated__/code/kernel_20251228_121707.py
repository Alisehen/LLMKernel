import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def conv_transpose2d_kernel(
    x_ptr,        # (N, C_in, H_in, W_in)
    w_ptr,        # (C_in, C_out_per_group, KH, KW)
    b_ptr,        # (C_out,) or dummy
    out_ptr,      # (N, C_out, H_out, W_out)
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    STRIDE_H, STRIDE_W,
    PADDING_H, PADDING_W,
    DILATION_H, DILATION_W,
    C_IN_PER_GROUP: tl.constexpr,
    C_OUT_PER_GROUP: tl.constexpr,
    TOTAL_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMENTS

    # Map linear index -> (n, co, ho, wo)
    HW_OUT = H_OUT * W_OUT
    C_HW_OUT = C_OUT * HW_OUT

    n = offs // C_HW_OUT
    rem1 = offs - n * C_HW_OUT
    co = rem1 // HW_OUT
    rem2 = rem1 - co * HW_OUT
    ho = rem2 // W_OUT
    wo = rem2 - ho * W_OUT

    # Grouped channels
    group = co // C_OUT_PER_GROUP
    co_in_group = co - group * C_OUT_PER_GROUP
    c_in_base = group * C_IN_PER_GROUP  # per-lane base input channel for this group

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over kernel spatial dims and input channels in the group
    for kh in range(KH):
        # vertical mapping: ho + pad_h = hi * stride_h + kh * dilation_h
        num_h = ho + PADDING_H - kh * DILATION_H
        hi = num_h // STRIDE_H
        rem_h = num_h % STRIDE_H
        mask_h = (num_h >= 0) & (rem_h == 0) & (hi < H_IN)

        for kw in range(KW):
            # horizontal mapping: wo + pad_w = wi * stride_w + kw * dilation_w
            num_w = wo + PADDING_W - kw * DILATION_W
            wi = num_w // STRIDE_W
            rem_w = num_w % STRIDE_W
            mask_w = (num_w >= 0) & (rem_w == 0) & (wi < W_IN)

            pix_mask = mask & mask_h & mask_w

            # Precompute kernel offset within (KH, KW)
            k_offset = kh * KW + kw

            for c_in_g in range(C_IN_PER_GROUP):
                ci = c_in_base + c_in_g  # (per-lane) input channel index

                # Input index: ((n*C_IN + ci)*H_IN + hi)*W_IN + wi
                inp_index = ((n * C_IN + ci) * H_IN + hi) * W_IN + wi

                # Weight index: ((ci*C_OUT_PER_GROUP + co_in_group)*KH + kh)*KW + kw
                w_index = ((ci * C_OUT_PER_GROUP + co_in_group) * KH + kh) * KW + kw

                # Safe load: input masked by valid-pixel mask
                x_val = tl.load(x_ptr + inp_index, mask=pix_mask, other=0.0)
                # Weight load masked only by "mask" to avoid OOB at tail elements
                w_val = tl.load(w_ptr + w_index, mask=mask, other=0.0)

                x_val_f32 = x_val.to(tl.float32)
                w_val_f32 = w_val.to(tl.float32)
                acc += x_val_f32 * w_val_f32

    # Add bias if present: bias depends only on output channel
    if HAS_BIAS:
        b_val = tl.load(b_ptr + co, mask=mask, other=0.0)
        acc += b_val.to(tl.float32)

    # Store result
    tl.store(out_ptr + offs, acc, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: tuple,
                            padding: tuple,
                            dilation: tuple,
                            groups: int) -> torch.Tensor:
    # Ensure CUDA + contiguous
    assert x.is_cuda and weight.is_cuda, "Input and weights must be CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, KH, KW = weight.shape
    assert C_in_w == C_in, "Weight C_in mismatch"

    C_out = C_out_per_group * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    total_elements = N * C_out * H_out * W_out
    BLOCK_SIZE = 128  # power-of-2, good trade-off for occupancy

    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

    conv_transpose2d_kernel[grid](
        x, weight, bias if bias is not None else out, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        C_IN_PER_GROUP=C_in // groups,
        C_OUT_PER_GROUP=C_out_per_group,
        TOTAL_ELEMENTS=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        KH=KH,
        KW=KW,
        HAS_BIAS=(bias is not None),
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for nn.ConvTranspose2d with arbitrary stride, padding,
    dilation and groups.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        # Use PyTorch module only to hold parameters (weight/bias, groups, etc.)
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding
        dilation = self.conv_transpose2d.dilation
        groups = self.conv_transpose2d.groups
        # Ensure running on CUDA for Triton
        return triton_conv_transpose2d(x, w, b, stride, padding, dilation, groups)
