import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fwd_input_kernel(
    x_ptr,            # [N, C_in, D_in, H_in, W_in] - contiguous
    w_ptr,            # [C_in, C_out_per_group, Kd, Kh, Kw] - contiguous
    y_ptr,            # [N, C_out, D_out, H_out, W_out] - contiguous
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
    n_elements_in,    # total elements in x
    C_in_per_group: tl.constexpr,
    C_out_per_group: tl.constexpr,
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward-mapping, input-stationary 3D transposed convolution.

    Primary loop: over input elements (n, c_in, d_in, h_in, w_in).
    For each input element, we compute its contributing output positions via:
        d_out = d_in * stride_d - padding_d + kd
        h_out = h_in * stride_h - padding_h + kh
        w_out = w_in * stride_w - padding_w + kw
    and scatter-add into y using atomics.
    """

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements_in

    # Decode flattened input index -> (n, c_in, d_in, h_in, w_in)
    # x is [N, C_in, D_in, H_in, W_in] and contiguous, so `offs` already matches
    # its flattened layout, but we still decode for coordinate math.
    w_in_idx = offs % W_in
    tmp = offs // W_in

    h_in_idx = tmp % H_in
    tmp = tmp // H_in

    d_in_idx = tmp % D_in
    tmp = tmp // D_in

    c_in_idx = tmp % C_in
    n_idx = tmp // C_in

    # Load input values (one per lane); x is contiguous so pointer = x_ptr + offs
    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Precompute constants for weight indexing
    w_cin_stride = C_out_per_group * Kd * Kh * Kw
    w_cout_stride = Kd * Kh * Kw
    w_kd_stride = Kh * Kw
    w_kh_stride = Kw

    # Group index from input channel: which output channels this input connects to
    group_idx = c_in_idx // C_in_per_group
    c_out_base = group_idx * C_out_per_group

    # For each kernel depth position
    for kd in range(Kd):
        d_out = d_in_idx * stride_d - padding_d + kd
        cond_d = (d_out >= 0) & (d_out < D_out)

        # For each kernel height position
        for kh in range(Kh):
            h_out = h_in_idx * stride_h - padding_h + kh
            cond_h = (h_out >= 0) & (h_out < H_out)

            # For each kernel width position
            for kw in range(Kw):
                w_out = w_in_idx * stride_w - padding_w + kw
                cond_w = (w_out >= 0) & (w_out < W_out)

                valid_spatial = mask & cond_d & cond_h & cond_w

                # If no lanes are valid for this (kd,kh,kw), we can still rely on
                # masking; Triton will ignore masked lanes.
                # Compute base offset in weight for this (kd,kh,kw), independent of c_in
                w_k_offset = kd * w_kd_stride + kh * w_kh_stride + kw

                # For each output channel within the group
                for c_out_rel in range(C_out_per_group):
                    c_out_idx = c_out_base + c_out_rel

                    # Weight index: [c_in, c_out_rel, kd, kh, kw]
                    w_idx = (
                        c_in_idx * w_cin_stride
                        + c_out_rel * w_cout_stride
                        + w_k_offset
                    )

                    w_val = tl.load(w_ptr + w_idx, mask=valid_spatial, other=0.0)

                    # Compute flattened output index for y:
                    # y[n, c_out, d_out, h_out, w_out] in contiguous layout
                    idx_y = (
                        (((n_idx * C_out + c_out_idx) * D_out + d_out) * H_out + h_out)
                        * W_out
                        + w_out
                    )

                    # Contribution from this input and kernel position
                    update = x_val * w_val

                    # Atomic add into output (scatter)
                    tl.atomic_add(y_ptr + idx_y, update, mask=valid_spatial)


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

    # Currently optimized for float32
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, \
        "This Triton kernel currently supports float32 tensors only."

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_pg, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in, "Weight C_in dimension must match input C_in"
    C_out = C_out_pg * groups

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    # Output shape follows PyTorch ConvTranspose3d formula (dilation=1)
    D_out = (D_in - 1) * stride_d - 2 * pad_d + Kd + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + Kh + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + Kw + out_pad_w

    y = torch.zeros(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    n_elements_in = x.numel()
    if n_elements_in == 0:
        # Only bias (if any) contributes
        if bias is not None:
            y += bias.view(1, C_out, 1, 1, 1)
        return y

    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups

    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(n_elements_in, meta["BLOCK_SIZE"]),)

    conv_transpose3d_fwd_input_kernel[grid](
        x,
        weight,
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
        n_elements_in,
        C_in_per_group=C_in_per_group,
        C_out_per_group=C_out_per_group,
        Kd=Kd,
        Kh=Kh,
        Kw=Kw,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    # Add bias (if present) after convolution
    if bias is not None:
        bias = bias.contiguous()
        y += bias.view(1, C_out, 1, 1, 1)

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
        # Use a real ConvTranspose3d module for parameter storage & CPU fallback
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
        # CPU or non-CUDA fallback: use PyTorch implementation directly
        if not x.is_cuda:
            return self.conv_transpose3d(x)

        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        output_padding = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups

        return triton_conv_transpose3d(x, w, b, stride, padding, output_padding, groups)
