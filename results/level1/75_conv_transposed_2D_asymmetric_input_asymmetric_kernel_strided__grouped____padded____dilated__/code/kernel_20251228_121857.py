import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,         # float32[N, C_in, H_in, W_in]
    w_ptr,         # float32[C_in, C_out_per_group, kH, kW]
    b_ptr,         # float32[C_out] or dummy
    out_ptr,       # float32[N, C_out, H_out, W_out]
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    kH,
    kW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    groups,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nh = tl.program_id(axis=0)
    pid_wo = tl.program_id(axis=1)

    # Decode n, oy from pid_nh (0 .. N*H_out-1)
    n = pid_nh // H_out
    oy = pid_nh - n * H_out

    # Flatten (oc, ox) into linear index along C_out * W_out
    lin_start = pid_wo * BLOCK_SIZE
    offs = lin_start + tl.arange(0, BLOCK_SIZE)
    total_owc = C_out * W_out
    mask = offs < total_owc

    oc = offs // W_out
    ox = offs - oc * W_out

    # Guard against oc out of range (for safety)
    mask = mask & (oc < C_out)

    # Group info
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    oc_group = oc // C_out_per_group          # [BLOCK]
    oc_in_group = oc - oc_group * C_out_per_group  # [BLOCK]

    # Initialize accumulator, optionally with bias
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if has_bias:
        bias_val = tl.load(b_ptr + oc, mask=mask, other=0.0)
        acc = bias_val

    # Main convolution transpose loops
    ky = 0
    while ky < kH:
        # Compute input y-coordinate from output oy and kernel row ky
        numerator_y = oy + pad_h - ky * dil_h
        iy = numerator_y // stride_h
        rem_y = numerator_y - iy * stride_h
        valid_y = (rem_y == 0) & (iy >= 0) & (iy < H_in)

        kx = 0
        while kx < kW:
            # Compute input x-coordinates (vectorized over ox)
            numerator_x = ox + pad_w - kx * dil_w
            ix = numerator_x // stride_w
            rem_x = numerator_x - ix * stride_w
            valid_x = (rem_x == 0) & (ix >= 0) & (ix < W_in)

            ic = 0
            while ic < C_in:
                ic_group = ic // C_in_per_group
                same_group = oc_group == ic_group

                # Combined mask for this (ic, ky, kx) contribution
                mask_iter = mask & valid_x & valid_y & same_group

                # Input index: ((n*C_in + ic)*H_in + iy)*W_in + ix
                base_in = ((n * C_in + ic) * H_in + iy) * W_in
                in_offset = base_in + ix

                # Weight index:
                # w_shape = [C_in, C_out_per_group, kH, kW]
                # idx = (((ic * C_out_per_group) + oc_in_group)*kH + ky)*kW + kx
                idx_w = (ic * C_out_per_group + oc_in_group) * (kH * kW) + ky * kW + kx

                x_val = tl.load(x_ptr + in_offset, mask=mask_iter, other=0.0)
                w_val = tl.load(w_ptr + idx_w, mask=mask_iter, other=0.0)

                acc += x_val * w_val
                ic += 1

            kx += 1
        ky += 1

    # Store result
    # out index: ((n*C_out + oc)*H_out + oy)*W_out + ox
    base_out = ((n * C_out) * H_out + oy) * W_out
    out_offset = base_out + ox
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: tuple,
                            padding: tuple,
                            dilation: tuple,
                            groups: int) -> torch.Tensor:
    """
    x:      [N, C_in, H_in, W_in]
    weight: [C_in, C_out_per_group, kH, kW]  (ConvTranspose2d layout)
    bias:   [C_out] or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, kH, kW = weight.shape
    assert C_in_w == C_in
    C_out = C_out_per_group * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + 1

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 128
    grid = lambda meta: (
        N * H_out,
        triton.cdiv(C_out * W_out, meta["BLOCK_SIZE"]),
    )

    has_bias = 1 if bias is not None else 0
    b_ptr = bias if bias is not None else weight  # dummy pointer when no bias

    conv_transpose2d_kernel[grid](
        x,
        weight,
        b_ptr,
        out,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        has_bias=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated replacement for nn.ConvTranspose2d with the same constructor.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
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
        return triton_conv_transpose2d(
            x,
            w,
            b,
            stride=self.conv_transpose2d.stride,
            padding=self.conv_transpose2d.padding,
            dilation=self.conv_transpose2d.dilation,
            groups=self.conv_transpose2d.groups,
        )
