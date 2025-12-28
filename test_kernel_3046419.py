import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_fwd_kernel(
    x_ptr,          # float32/float16 [N, C, H, W]
    w_ptr,          # float32/float16 [C, 1, K, K]
    b_ptr,          # float32/float16 [C]
    y_ptr,          # float32/float16 [N, C, H_out, W_out]
    N, C, H, W,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(axis=0)  # over N * C
    pid_ho = tl.program_id(axis=1)  # over H_out tiles
    pid_wo = tl.program_id(axis=2)  # over W_out tiles

    # Decompose (n, c) from flattened pid_nc
    n = pid_nc // C
    c = pid_nc % C

    # Spatial tile coordinates
    ho_start = pid_ho * BLOCK_H
    wo_start = pid_wo * BLOCK_W

    ho_offsets = ho_start + tl.arange(0, BLOCK_H)
    wo_offsets = wo_start + tl.arange(0, BLOCK_W)

    ho = ho_offsets[:, None]  # [BH, 1]
    wo = wo_offsets[None, :]  # [1, BW]

    # Mask for valid output coordinates
    mask_hw = (ho < H_out) & (wo < W_out)

    # Base offsets for this (n, c) in input and output
    nc = n * C + c
    base_in_nc = nc * H * W
    base_out_nc = nc * H_out * W_out

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Depthwise convolution: each channel has its own KxK kernel
    for kh in range(KERNEL_SIZE):
        for kw in range(KERNEL_SIZE):
            h_in = ho * stride_h - pad_h + kh
            w_in = wo * stride_w - pad_w + kw

            in_bounds = (
                (h_in >= 0) & (h_in < H) &
                (w_in >= 0) & (w_in < W) &
                mask_hw
            )

            in_offsets = base_in_nc + h_in * W + w_in
            x_vals = tl.load(x_ptr + in_offsets, mask=in_bounds, other=0.0)

            w_offset = c * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw
            w_val = tl.load(w_ptr + w_offset)

            acc += x_vals.to(tl.float32) * w_val.to(tl.float32)

    # Add bias per channel
    b_val = tl.load(b_ptr + c)
    acc = acc + b_val.to(tl.float32)

    out_offsets = base_out_nc + ho * W_out + wo
    tl.store(y_ptr + out_offsets, acc, mask=mask_hw)


def triton_depthwise_conv2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride=1,
                            padding=0) -> torch.Tensor:
    """
    Depthwise 2D convolution (NCHW) implemented with Triton.
    Expects weight from nn.Conv2d with groups=in_channels: [C, 1, K, K].
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernel requires CUDA tensors"

    N, C, H, W = x.shape
    Cout, Cin_group, Kh, Kw = weight.shape
    assert Cout == C and Cin_group == 1, "Weight must be [C, 1, K, K]"
    assert Kh == Kw, "Kernel must be square"

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    H_out = (H + 2 * pad_h - Kh) // stride_h + 1
    W_out = (W + 2 * pad_w - Kw) // stride_w + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    # Ensure bias tensor
    if bias is None:
        bias_t = torch.zeros(C, device=x.device, dtype=x.dtype)
    else:
        bias_t = bias

    BLOCK_H = 16
    BLOCK_W = 16

    def grid(meta):
        return (
            N * C,
            triton.cdiv(H_out, meta['BLOCK_H']),
            triton.cdiv(W_out, meta['BLOCK_W']),
        )

    depthwise_conv2d_fwd_kernel[grid](
        x, weight, bias_t, y,
        N, C, H, W,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        KERNEL_SIZE=Kh,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Depthwise 2D convolution using a high-performance Triton kernel.
    Interface matches the original Model.
    """
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        # Use nn.Conv2d only for parameter storage (weights/bias)
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch on CPU since Triton requires CUDA
        if not x.is_cuda:
            return self.conv2d(x)

        return triton_depthwise_conv2d(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
        )
