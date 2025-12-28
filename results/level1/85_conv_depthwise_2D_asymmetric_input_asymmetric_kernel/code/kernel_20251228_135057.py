import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,          # float* [N, C, H, W]
    w_ptr,          # float* [C, 1, K_H, K_W]
    b_ptr,          # float* [C] (ignored if has_bias=False)
    y_ptr,          # float* [N, C, H_out, W_out]
    N, C, H, W,
    H_out, W_out,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    has_bias: tl.constexpr,
    K_H: tl.constexpr,  # kernel height
    K_W: tl.constexpr,  # kernel width
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offs = block_start + tl.arange(0, BLOCK)
    mask = offs < (N * C * H_out * W_out)

    # Decode flattened index -> (n, c, h_out, w_out)
    w_out_idx = offs % W_out
    tmp = tl.floor_div(offs, W_out)
    h_out_idx = tmp % H_out
    tmp = tl.floor_div(tmp, H_out)
    c_idx = tmp % C
    n_idx = tl.floor_div(tmp, C)

    # Base offsets for this (n, c) plane
    # input:  ((n * C + c) * H + h) * W + w
    # output: ((n * C + c) * H_out + h_out) * W_out + w_out
    nc = n_idx * C + c_idx
    in_plane_offset = nc * H * W
    out_plane_offset = nc * H_out * W_out

    h_out = h_out_idx
    w_out = w_out_idx

    # Top-left input coordinate for this output position
    h_in_origin = h_out * stride_h - padding_h
    w_in_origin = w_out * stride_w - padding_w

    # Accumulator in fp32 for better precision
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # Depthwise convolution: loop over kernel spatial dims
    for kh in tl.static_range(0, K_H):
        h_in = h_in_origin + kh * dilation_h

        # Clamp for safe addressing; we'll mask out of bounds later
        h_in_clamped = tl.maximum(0, tl.minimum(h_in, H - 1))

        for kw in tl.static_range(0, K_W):
            w_in = w_in_origin + kw * dilation_w

            w_in_clamped = tl.maximum(0, tl.minimum(w_in, W - 1))

            in_bounds = (
                (h_in >= 0)
                & (h_in < H)
                & (w_in >= 0)
                & (w_in < W)
                & mask
            )

            in_offsets = in_plane_offset + h_in_clamped * W + w_in_clamped
            x_val = tl.load(x_ptr + in_offsets, mask=in_bounds, other=0.0)
            x_val = x_val.to(tl.float32)

            # Weight index: [C, 1, K_H, K_W] -> c * K_H * K_W + kh * K_W + kw
            w_offsets = c_idx * (K_H * K_W) + kh * K_W + kw
            w_val = tl.load(w_ptr + w_offsets, mask=mask, other=0.0)
            w_val = w_val.to(tl.float32)

            acc += x_val * w_val

    if has_bias:
        # Bias per channel: [C]
        b_val = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
        b_val = b_val.to(tl.float32)
        acc += b_val

    out_offsets = out_plane_offset + h_out * W_out + w_out
    tl.store(y_ptr + out_offsets, acc, mask=mask)


def depthwise_conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride_h: int,
    stride_w: int,
    padding_h: int,
    padding_w: int,
    dilation_h: int,
    dilation_w: int,
) -> torch.Tensor:
    """
    Depthwise 2D convolution (groups == in_channels) implemented with Triton.
    x: [N, C, H, W], weight: [C, 1, K_H, K_W], bias: [C] or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.is_contiguous(), "Input must be contiguous NCHW"
    assert weight.is_contiguous(), "Weight must be contiguous"

    N, C, H, W = x.shape
    C_w, one, K_H, K_W = weight.shape
    assert one == 1, "Weight shape for depthwise must be [C, 1, K_H, K_W]"
    assert C_w == C, "Weight C dimension must match input channels"

    # Output spatial dimensions (PyTorch Conv2d formula)
    H_out = (H + 2 * padding_h - dilation_h * (K_H - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_w - dilation_w * (K_W - 1) - 1) // stride_w + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    total_elems = N * C * H_out * W_out
    BLOCK = 256

    grid = lambda meta: (triton.cdiv(max(total_elems, 1), meta["BLOCK"]),)

    has_bias = bias is not None
    b_tensor = bias if bias is not None else weight  # dummy pointer if no bias

    depthwise_conv2d_kernel[grid](
        x,
        weight,
        b_tensor,
        y,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        has_bias=has_bias,
        K_H=K_H,
        K_W=K_W,
        BLOCK=BLOCK,
    )

    return y


class ModelNew(nn.Module):
    """
    Depthwise 2D convolution implemented with Triton, parameter-compatible with the original Model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Keep the same module / parameter structure as the original Model
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,  # depthwise: out_channels == in_channels
            (kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
            padding=(padding_h, padding_w),
            dilation=(dilation_h, dilation_w),
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel with Conv2d weights/bias
        assert x.dim() == 4, "Expected input of shape [N, C, H, W]"
        x = x.contiguous()
        w = self.conv2d.weight
        b = self.conv2d.bias

        stride_h, stride_w = self.conv2d.stride
        padding_h, padding_w = self.conv2d.padding
        dilation_h, dilation_w = self.conv2d.dilation

        return depthwise_conv2d_triton(
            x,
            w,
            b,
            stride_h=stride_h,
            stride_w=stride_w,
            padding_h=padding_h,
            padding_w=padding_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )
