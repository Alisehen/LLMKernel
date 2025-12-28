import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,          # float32[N, C_in, H_in, W_in]
    w_ptr,          # float32[C_in, C_out_per_g, kH, kW]
    b_ptr,          # float32[C_out]  (ignored if has_bias=False)
    y_ptr,          # float32[N, C_out, H_out, W_out]
    N, C_in,
    H_in, W_in,
    C_out,
    H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    C_in_per_g,
    C_out_per_g,
    n_elements,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decompose linear index -> (n, co, ho, wo) for each lane
    tmp = offs
    # wo
    tmp_w = tmp // W_out
    wo = tmp - tmp_w * W_out
    tmp = tmp_w
    # ho
    tmp_h = tmp // H_out
    ho = tmp - tmp_h * H_out
    tmp = tmp_h
    # co
    tmp_c = tmp // C_out
    co = tmp - tmp_c * C_out
    n = tmp_c

    # Initialize accumulator
    out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Optional bias
    if HAS_BIAS:
        bias_val = tl.load(b_ptr + co, mask=mask, other=0.0)
        out = out + bias_val

    kernel_area = kH * kW

    # Group handling
    g_out = co // C_out_per_g          # [BLOCK_SIZE]
    oc_in_group = co - g_out * C_out_per_g  # [BLOCK_SIZE]

    # Iterate over input channels
    for ic in range(0, C_in):
        g_in = ic // C_in_per_g  # python int
        same_group = g_out == g_in

        ic_base = ic * C_out_per_g
        w_idx0 = (ic_base + oc_in_group) * kernel_area  # [BLOCK_SIZE]

        # For each kernel position
        for kh in range(0, kH):
            # compute corresponding input h index
            t_h = ho + pad_h - kh * dil_h
            # range where a corresponding input row may exist
            within_h = (t_h >= 0) & (t_h <= (H_in - 1) * stride_h)
            hi = t_h // stride_h
            mod_h = t_h - hi * stride_h
            valid_h = within_h & (mod_h == 0)
            hi = tl.where(valid_h, hi, 0)

            for kw in range(0, kW):
                t_w = wo + pad_w - kw * dil_w
                within_w = (t_w >= 0) & (t_w <= (W_in - 1) * stride_w)
                wi = t_w // stride_w
                mod_w = t_w - wi * stride_w
                valid_w = within_w & (mod_w == 0)
                wi = tl.where(valid_w, wi, 0)

                full_mask = mask & same_group & valid_h & valid_w

                # input index: ((n*C_in + ic)*H_in + hi)*W_in + wi
                idx_x = ((n * C_in + ic) * H_in + hi) * W_in + wi

                # weight index: (ic*C_out_per_g + oc_in_group)*kH*kW + kh*kW + kw
                kw_flat = kh * kW + kw
                idx_w = w_idx0 + kw_flat

                x_val = tl.load(x_ptr + idx_x, mask=full_mask, other=0.0)
                w_val = tl.load(w_ptr + idx_w, mask=full_mask, other=0.0)
                out = out + x_val * w_val

    # Store result
    tl.store(y_ptr + offs, out, mask=mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """
    x:       (N, C_in, H_in, W_in)
    weight:  (C_in, C_out/groups, kH, kW)  -- nn.ConvTranspose2d layout
    bias:    (C_out,) or None
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_g, kH, kW = weight.shape
    assert C_in_w == C_in, "weight C_in mismatch with input"
    C_out = C_out_per_g * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    opad_h, opad_w = output_padding
    dil_h, dil_w = dilation

    # Output shape as in PyTorch
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + opad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + opad_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    n_elements = N * C_out * H_out * W_out
    if n_elements == 0:
        return y

    C_in_per_g = C_in // groups
    C_out_per_g = C_out // groups

    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    has_bias = bias is not None
    b_tensor = bias if has_bias else x  # dummy pointer if no bias

    conv_transpose2d_kernel[grid](
        x, weight, b_tensor, y,
        N, C_in,
        H_in, W_in,
        C_out,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        C_in_per_g,
        C_out_per_g,
        n_elements,
        HAS_BIAS=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Transposed 2D convolution implemented with a high-performance Triton kernel.
    API-compatible with the given PyTorch Model.
    """

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
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding
        output_padding = self.conv_transpose2d.output_padding
        dilation = self.conv_transpose2d.dilation
        groups = self.conv_transpose2d.groups

        return triton_conv_transpose2d(
            x, w, b, stride, padding, output_padding, dilation, groups
        )
