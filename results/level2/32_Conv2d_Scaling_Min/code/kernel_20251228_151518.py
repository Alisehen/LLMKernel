import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    scale,
    BLOCK_M: tl.constexpr,  # number of (b, y, x) points per program
    BLOCK_CO: tl.constexpr,  # number of output channels processed per block
):
    pid_m = tl.program_id(0)

    # Total number of output spatial points (B * Ho * Wo)
    N_points = B * Ho * Wo

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_points

    # Decode linear index into (b, y, x)
    hw = Ho * Wo
    b = offs_m // hw
    rem = offs_m % hw
    y = rem // Wo
    x = rem % Wo

    # Initialize running minimum (per output spatial location)
    # Large positive value as initial min
    min_val = tl.full((BLOCK_M,), 3.4e38, tl.float32)

    # Loop over output channels in tiles of size BLOCK_CO
    for oc_start in range(0, Co, BLOCK_CO):
        oc_offsets = oc_start + tl.arange(0, BLOCK_CO)
        mask_co = oc_offsets < Co

        # Accumulator for this output-channel tile
        acc = tl.zeros((BLOCK_M, BLOCK_CO), dtype=tl.float32)

        # Convolution: sum over input channels and kernel spatial dims
        # No padding, stride=1, dilation=1 assumed.
        for ic in range(0, Ci):
            for ky in range(0, Kh):
                for kx in range(0, Kw):
                    iy = y + ky
                    ix = x + kx
                    # Within valid region by construction for standard conv
                    x_ptrs = (
                        x_ptr
                        + b * stride_xb
                        + ic * stride_xc
                        + iy * stride_xh
                        + ix * stride_xw
                    )
                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)

                    w_ptrs = (
                        w_ptr
                        + oc_offsets * stride_wco
                        + ic * stride_wci
                        + ky * stride_wkh
                        + kx * stride_wkw
                    )
                    w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0)

                    acc += x_vals[:, None] * w_vals[None, :]

        # Add bias for this output-channel tile
        bias_vals = tl.load(bias_ptr + oc_offsets, mask=mask_co, other=0.0)
        acc += bias_vals[None, :]

        # Scale
        acc = acc * scale

        # Reduce over output-channel tile (per spatial point)
        tile_min = tl.min(acc, 1)
        min_val = tl.minimum(min_val, tile_min)

    # Store result: output shape [B, 1, Ho, Wo]
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + 0 * stride_oc
        + y * stride_oh
        + x * stride_ow
    )
    tl.store(out_ptrs, min_val, mask=mask_m)


def conv_scale_min_triton(x, weight, bias, scale_factor):
    """
    x:       [B, Ci, Hi, Wi]
    weight:  [Co, Ci, Kh, Kw]
    bias:    [Co]
    Returns: [B, 1, Ho, Wo] with Ho/Wo as in PyTorch conv2d (stride=1, padding=0, dilation=1).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    B, Ci, Hi, Wi = x.shape
    Co, Ci_w, Kh, Kw = weight.shape
    assert Ci == Ci_w, "In-channel mismatch between input and weight"
    assert bias.shape[0] == Co, "Bias must match out_channels"

    # Conv2d output spatial size for stride=1, padding=0, dilation=1
    Ho = Hi - Kh + 1
    Wo = Wi - Kw + 1
    assert Ho > 0 and Wo > 0, "Invalid kernel size for given input"

    out = torch.empty((B, 1, Ho, Wo), device=x.device, dtype=x.dtype)

    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wco, stride_wci, stride_wkh, stride_wkw = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()

    N_points = B * Ho * Wo

    def grid(meta):
        return (triton.cdiv(N_points, meta["BLOCK_M"]),)

    conv_scale_min_kernel[grid](
        x, weight, bias, out,
        B, Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wco, stride_wci, stride_wkh, stride_wkw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        scale_factor,
        BLOCK_M=64,
        BLOCK_CO=32,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
        Conv2d -> scale -> min over channel dim (keepdim=True)
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        return conv_scale_min_triton(x, self.weight, self.bias, self.scale_factor)
