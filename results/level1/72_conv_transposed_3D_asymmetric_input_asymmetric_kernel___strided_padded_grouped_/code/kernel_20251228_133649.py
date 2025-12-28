import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fwd_kernel(
    x_ptr,            # [N, C_in, D_in, H_in, W_in]
    w_ptr,            # [C_in, C_out_per_group, Kd, Kh, Kw]
    b_ptr,            # [C_out] or dummy
    y_ptr,            # [N, C_out, D_out, H_out, W_out]
    N,
    C_in,
    C_out,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    stride_d,
    stride_h,
    stride_w,
    padding_d,
    padding_h,
    padding_w,
    groups,
    n_elements,       # total elements in y
    C_in_per_group: tl.constexpr,
    C_out_per_group: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decode flattened index -> (n, c_out, d_out, h_out, w_out)
    w_out = offs % W_out
    tmp = offs // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c_out = tmp % C_out
    n = tmp // C_out

    # Initialize accumulator (optionally with bias)
    if has_bias:
        acc = tl.load(b_ptr + c_out, mask=mask, other=0.0)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Group and channel mapping
    group_idx = c_out // C_out_per_group
    c_out_rel = c_out - group_idx * C_out_per_group
    c_in_start = group_idx * C_in_per_group

    # Loop over input channels within the group and kernel elements
    for cin_g in range(C_in_per_group):
        c_in = c_in_start + cin_g

        for kd in range(Kd):
            d_in_nom = d_out + padding_d - kd
            d_div = d_in_nom // stride_d
            cond_d = (d_in_nom % stride_d == 0) & (d_div >= 0) & (d_div < D_in)

            for kh in range(Kh):
                h_in_nom = h_out + padding_h - kh
                h_div = h_in_nom // stride_h
                cond_h = (h_in_nom % stride_h == 0) & (h_div >= 0) & (h_div < H_in)

                for kw in range(Kw):
                    w_in_nom = w_out + padding_w - kw
                    w_div = w_in_nom // stride_w
                    cond_w = (w_in_nom % stride_w == 0) & (w_div >= 0) & (w_div < W_in)

                    valid = mask & cond_d & cond_h & cond_w

                    # Compute input index
                    x_index = (
                        (((n * C_in + c_in) * D_in + d_div) * H_in + h_div) * W_in
                        + w_div
                    )

                    x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)

                    # Weight index (same for all elements in the block)
                    w_offset = (
                        (((c_in) * C_out_per_group + c_out_rel) * Kd + kd) * Kh * Kw
                        + kh * Kw
                        + kw
                    )
                    w_val = tl.load(w_ptr + w_offset)

                    acc += x_val * w_val

    # Store result
    tl.store(y_ptr + offs, acc, mask=mask)


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding, groups):
    """
    x:       [N, C_in, D_in, H_in, W_in] (CUDA, contiguous)
    weight:  [C_in, C_out/groups, Kd, Kh, Kw] (CUDA, contiguous)
    bias:    [C_out] or None
    stride, padding, output_padding: 3-tuples
    groups:  int
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_pg, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in
    C_out = C_out_pg * groups

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w

    y = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    n_elements = y.numel()
    if n_elements == 0:
        return y

    if bias is None:
        bias = x.new_zeros(C_out)
        has_bias = False
    else:
        bias = bias.contiguous()
        has_bias = True

    BLOCK_SIZE = 256
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    conv_transpose3d_fwd_kernel[grid](
        x,
        weight,
        bias,
        y,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        groups,
        n_elements,
        C_in_per_group=C_in_per_group,
        C_out_per_group=C_out_per_group,
        Kd=Kd,
        Kh=Kh,
        Kw=Kw,
        has_bias=has_bias,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return y


class ModelNew(nn.Module):
    """
    3D transposed convolution implemented with a high-performance Triton kernel,
    parameter-compatible with nn.ConvTranspose3d.
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
    ):
        super().__init__()
        # Keep a real ConvTranspose3d module for parameter storage & CPU fallback
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
        # CPU or non-CUDA fallback: use PyTorch implementation
        if not x.is_cuda:
            return self.conv_transpose3d(x)

        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        output_padding = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups

        return triton_conv_transpose3d(x, w, b, stride, padding, output_padding, groups)
