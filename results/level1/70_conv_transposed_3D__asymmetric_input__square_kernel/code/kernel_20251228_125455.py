import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,          # float32[N, Cin, Di, Hi, Wi]
    w_ptr,          # float32[Cin, Cout, kD, kH, kW]
    b_ptr,          # float32[Cout]
    y_ptr,          # float32[N, Cout, Do, Ho, Wo]
    N, Cin, Cout,
    Di, Hi, Wi,
    Do, Ho, Wo,
    kD, kH, kW,
    BLOCK_M: tl.constexpr,  # tiles over (N, Do, Ho, Wo)
    BLOCK_C: tl.constexpr,  # tiles over Cout
):
    pid_m = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    # Total number of output spatial positions per batch: Do * Ho * Wo
    s_do = Ho * Wo
    s_spatial = Do * s_do
    total_positions = N * s_spatial

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_m = offs_m < total_positions
    mask_c = offs_c < Cout

    # Decode (n, z, y, x) from flattened position offs_m
    pos = offs_m
    x = pos % Wo
    pos = pos // Wo
    y = pos % Ho
    pos = pos // Ho
    z = pos % Do
    n = pos // Do

    # Input strides (N, Cin, Di, Hi, Wi) in elements
    sW_in = 1
    sH_in = Wi
    sD_in = Hi * Wi
    sC_in = Di * sD_in
    sN_in = Cin * sC_in

    # Output strides (N, Cout, Do, Ho, Wo) in elements
    sW_out = 1
    sH_out = Wo
    sD_out = Ho * Wo
    sC_out = Do * sD_out
    sN_out = Cout * sC_out

    # Weight layout: (Cin, Cout, kD, kH, kW), contiguous
    K = kD * kH * kW
    # Each (ci, co) pair has a contiguous block of size K

    # Accumulator tile: [BLOCK_M, BLOCK_C]
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    # Loop over input channels and kernel spatial positions
    for ci in range(0, Cin):
        ci_val = tl.full((), ci, tl.int32)

        ci_weight_base = ci * Cout * K
        for kz in range(0, kD):
            kz_val = tl.full((), kz, tl.int32)
            for ky in range(0, kH):
                ky_val = tl.full((), ky, tl.int32)
                for kx in range(0, kW):
                    kx_val = tl.full((), kx, tl.int32)

                    # Compute corresponding input coordinates for each output position
                    # For stride=1, padding=0, dilation=1, output_padding=0 (full convolution):
                    # iz = z - kz, iy = y - ky, ix = x - kx
                    iz = z - kz_val
                    iy = y - ky_val
                    ix = x - kx_val

                    in_bounds_z = (iz >= 0) & (iz < Di)
                    in_bounds_y = (iy >= 0) & (iy < Hi)
                    in_bounds_x = (ix >= 0) & (ix < Wi)
                    in_bounds = in_bounds_z & in_bounds_y & in_bounds_x & mask_m

                    # Avoid invalid memory access: clamp to 0 when out-of-bounds
                    iz_safe = tl.where(in_bounds, iz, 0)
                    iy_safe = tl.where(in_bounds, iy, 0)
                    ix_safe = tl.where(in_bounds, ix, 0)

                    # Input linear offsets for BLOCK_M positions
                    in_offsets = (
                        n * sN_in
                        + ci_val * sC_in
                        + iz_safe * sD_in
                        + iy_safe * sH_in
                        + ix_safe * sW_in
                    )

                    x_vec = tl.load(x_ptr + in_offsets, mask=in_bounds, other=0.0)

                    # Weight offsets for BLOCK_C output channels
                    k_idx = (kz * kH + ky) * kW + kx
                    w_offsets = ci_weight_base + offs_c * K + k_idx
                    w_vec = tl.load(w_ptr + w_offsets, mask=mask_c, other=0.0)

                    # Outer product update: [BLOCK_M, BLOCK_C]
                    acc += x_vec[:, None] * w_vec[None, :]

    # Add bias
    bias_vals = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0)
    acc += bias_vals[None, :]

    # Store results
    base_out = (
        n * sN_out
        + z * sD_out
        + y * sH_out
        + x * sW_out
    )  # [BLOCK_M]

    out_offsets = base_out[:, None] + offs_c[None, :] * sC_out
    out_mask = mask_m[:, None] & mask_c[None, :]
    tl.store(y_ptr + out_offsets, acc, mask=out_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fast path: ConvTranspose3d for stride=1, padding=0, dilation=1, output_padding=0, groups=1.

    x:      (N, Cin, Di, Hi, Wi), float32, CUDA, contiguous
    weight: (Cin, Cout, kD, kH, kW), float32, CUDA, contiguous
    bias:   (Cout,), float32, CUDA, contiguous
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32
    assert weight.dtype == torch.float32
    assert bias.dtype == torch.float32

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, Cin, Di, Hi, Wi = x.shape
    Cin_w, Cout, kD, kH, kW = weight.shape
    assert Cin == Cin_w, "Input channels and weight channels must match"

    # For stride=1, padding=0, dilation=1, output_padding=0:
    Do = Di + kD - 1
    Ho = Hi + kH - 1
    Wo = Wi + kW - 1

    y = torch.empty((N, Cout, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_C = 32

    total_positions = N * Do * Ho * Wo

    grid = lambda meta: (
        triton.cdiv(total_positions, meta["BLOCK_M"]),
        triton.cdiv(Cout, meta["BLOCK_C"]),
    )

    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        N, Cin, Cout,
        Di, Hi, Wi,
        Do, Ho, Wo,
        kD, kH, kW,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated ConvTranspose3d for the common case:
      stride=1, padding=0, dilation=1, output_padding=0, groups=1.

    Falls back to nn.ConvTranspose3d for other configurations or non-CUDA tensors.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Use PyTorch layer for parameter initialization and fallback
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Cache configuration
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conditions for Triton fast path
        use_triton = (
            x.is_cuda
            and self.groups == 1
            and self.stride == 1
            and self.padding == 0
            and self.output_padding == 0
            and self.dilation == 1
            and x.dtype == torch.float32
            and self.conv_transpose3d.weight.dtype == torch.float32
            and (self.conv_transpose3d.bias is None or self.conv_transpose3d.bias.dtype == torch.float32)
        )

        if use_triton:
            weight = self.conv_transpose3d.weight
            if self.conv_transpose3d.bias is None:
                # Create a zero bias on-the-fly (kept on the same device)
                bias = torch.zeros(
                    weight.shape[1],
                    device=weight.device,
                    dtype=weight.dtype,
                )
            else:
                bias = self.conv_transpose3d.bias
            return triton_conv_transpose3d(x, weight, bias)

        # Fallback: full-featured PyTorch implementation
        return self.conv_transpose3d(x)
