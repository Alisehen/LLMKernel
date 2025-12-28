import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,        # float32[N, Cin, H_in, W_in]
    w_ptr,        # float32[Cin, Cout, kH, kW]
    b_ptr,        # float32[Cout] or dummy
    out_ptr,      # float32[N, Cout, H_out, W_out]
    N, Cin, Cout,
    H_in, W_in,
    H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)
    pid_co = tl.program_id(axis=2)

    n = pid_n  # batch index

    # Offsets along output channels
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = co_offsets < Cout

    # Offsets along flattened spatial dimension (H_out * W_out)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H_out * W_out)

    # Compute (oh, ow) from flat offsets
    oh = hw_offsets // W_out
    ow = hw_offsets % W_out

    # Prepare accumulator [BLOCK_CO, BLOCK_HW]
    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # Precompute strides for weight [Cin, Cout, kH, kW]
    stride_w_cin = Cout * kH * kW
    stride_w_co = kH * kW
    stride_w_kh = kW
    # stride_w_kw = 1

    # Base offset for this batch in input and output tensors
    in_batch_stride = Cin * H_in * W_in
    out_batch_stride = Cout * H_out * W_out
    x_batch_ptr = x_ptr + n * in_batch_stride
    out_batch_ptr = out_ptr + n * out_batch_stride

    # Loop over input channels and kernel spatial dimensions
    for ci in range(0, Cin):
        x_ci_ptr = x_batch_ptr + ci * (H_in * W_in)

        for kh in range(0, kH):
            # Compute input height indices for all output positions in this tile
            h_in_numer = oh + pad_h - kh
            h_ge_zero = h_in_numer >= 0
            hi = h_in_numer // stride_h
            h_in_range = hi < H_in
            h_divisible = (h_in_numer % stride_h) == 0
            h_mask = h_ge_zero & h_in_range & h_divisible

            for kw in range(0, kW):
                # Compute input width indices
                w_in_numer = ow + pad_w - kw
                w_ge_zero = w_in_numer >= 0
                wi = w_in_numer // stride_w
                w_in_range = wi < W_in
                w_divisible = (w_in_numer % stride_w) == 0

                # Combined valid mask for input positions
                mask_in_hw = mask_hw & h_mask & w_ge_zero & w_in_range & w_divisible

                # Compute input offsets for this (ci, kh, kw) for all HW in tile
                x_hw_offsets = hi * W_in + wi
                x_ptr_offsets = x_ci_ptr + x_hw_offsets

                x_vals = tl.load(x_ptr_offsets, mask=mask_in_hw, other=0.0)  # [BLOCK_HW]
                x_tile = x_vals[None, :]  # [1, BLOCK_HW]

                # Compute weight offsets for this (ci, kh, kw) and CO tile
                w_offsets = (
                    ci * stride_w_cin
                    + co_offsets * stride_w_co
                    + kh * stride_w_kh
                    + kw
                )
                w_vals = tl.load(w_ptr + w_offsets, mask=mask_co, other=0.0)  # [BLOCK_CO]
                w_tile = w_vals[:, None]  # [BLOCK_CO, 1]

                # Outer-product accumulate
                acc += w_tile * x_tile

    # Add bias if present
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)  # [BLOCK_CO]
        b_tile = b_vals[:, None]  # [BLOCK_CO, 1]
        acc += b_tile

    # Store results
    out_hw_offsets = oh * W_out + ow  # [BLOCK_HW]
    out_offsets = co_offsets[:, None] * (H_out * W_out) + out_hw_offsets[None, :]
    store_mask = mask_co[:, None] & mask_hw[None, :]

    tl.store(out_batch_ptr + out_offsets, acc, mask=store_mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: tuple,
                            padding: tuple) -> torch.Tensor:
    """
    x:       [N, Cin, H_in, W_in]
    weight:  [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d layout)
    bias:    [Cout] or None
    stride:  (stride_h, stride_w)
    padding: (pad_h, pad_w)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernel."

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, Cin, H_in, W_in = x.shape
    Cin_w, Cout, kH, kW = weight.shape
    assert Cin_w == Cin, "Inconsistent in_channels between input and weight."

    stride_h, stride_w = int(stride[0]), int(stride[1])
    pad_h, pad_w = int(padding[0]), int(padding[1])

    # Standard ConvTranspose2d output shape (dilation=1, output_padding=0)
    H_out = (H_in - 1) * stride_h - 2 * pad_h + kH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + kW

    out = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_HW = 64  # must be power-of-two
    BLOCK_CO = 32  # must be power-of-two

    has_bias = bias is not None
    b_ptr = bias if has_bias else out.new_empty(1)

    grid = (
        N,
        triton.cdiv(H_out * W_out, BLOCK_HW),
        triton.cdiv(Cout, BLOCK_CO),
    )

    conv_transpose2d_kernel[grid](
        x, weight, b_ptr, out,
        N, Cin, Cout,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        HAS_BIAS=has_bias,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CO=BLOCK_CO,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated 2D transposed convolution, API-compatible with the given Model.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple = (1, 1),
                 padding: tuple = (0, 0),
                 bias: bool = False):
        super().__init__()
        # Keep a standard ConvTranspose2d module to own parameters/state_dict
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # Fallback to PyTorch implementation on CPU
            return self.conv_transpose2d(x)

        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding

        return triton_conv_transpose2d(x, w, b, stride, padding)
