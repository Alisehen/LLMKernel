# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_relu_double_bias_kernel(
    x_ptr,          # *const float
    w_ptr,          # *const float
    conv_bias_ptr,  # *const float  (Conv2d bias, length = OC)
    add_bias_ptr,   # *const float  (extra bias after ReLU, length = OC)
    y_ptr,          # *float
    N, C_in, H, W,
    OC, H_out, W_out,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_oc, stride_y_h, stride_y_w,
    K_H, K_W,
    P,                          # total number of output positions = N * H_out * W_out
    BLOCK_M: tl.constexpr,      # tile over output positions (N*H_out*W_out)
    BLOCK_N: tl.constexpr,      # tile over output channels
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] over output positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] over output channels

    mask_m = offs_m < P
    mask_n = offs_n < OC

    # Decode flattened output position index -> (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator for the output tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution: sum over input channels and kernel spatial dims
    for ic in range(0, C_in):
        for kh in range(0, K_H):
            for kw in range(0, K_W):
                in_h = oh_idx + kh
                in_w = ow_idx + kw

                x_ptrs = (
                    x_ptr
                    + n_idx * stride_x_n
                    + ic * stride_x_c
                    + in_h * stride_x_h
                    + in_w * stride_x_w
                )
                w_ptrs = (
                    w_ptr
                    + offs_n * stride_w_oc
                    + ic * stride_w_ic
                    + kh * stride_w_kh
                    + kw * stride_w_kw
                )

                x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)      # [BLOCK_M]
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)      # [BLOCK_N]

                acc += x_vals[:, None] * w_vals[None, :]

    # Add Conv2d's internal bias BEFORE ReLU: conv(x, w) + conv_bias
    conv_bias_vals = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += conv_bias_vals[None, :]

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Add extra bias AFTER ReLU (broadcast over spatial dims)
    add_bias_vals = tl.load(add_bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += add_bias_vals[None, :]

    # Store results
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_oc
        + oh_idx[:, None] * stride_y_h
        + ow_idx[:, None] * stride_y_w
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_conv_relu_double_bias(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    add_bias: torch.Tensor,
) -> torch.Tensor:
    """
    x:         [N, C_in, H, W]
    weight:    [OC, C_in, K_H, K_W]  (Conv2d weight)
    conv_bias: [OC]                  (Conv2d bias, applied before ReLU)
    add_bias:  [OC] or [OC, 1, 1]    (extra bias, applied after ReLU)

    Computes: ReLU(conv2d(x, weight) + conv_bias) + add_bias
    with stride=1, padding=0, dilation=1, groups=1.
    """
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda and add_bias.is_cuda, \
        "All tensors must be on CUDA"
    assert x.dtype == weight.dtype == conv_bias.dtype == add_bias.dtype == torch.float32, \
        "This implementation assumes float32 tensors"

    N, C_in, H, W = x.shape
    OC, Cw_in, K_H, K_W = weight.shape
    assert C_in == Cw_in, "Input channels must match weight's in_channels"
    assert conv_bias.numel() == OC, "conv_bias must be length = out_channels"

    # For this implementation: stride=1, padding=0, dilation=1
    H_out = H - K_H + 1
    W_out = W - K_W + 1
    assert H_out > 0 and W_out > 0, "Kernel size too large for input spatial dimensions"

    # Biases: flatten to [OC]
    conv_bias_flat = conv_bias.view(OC)
    add_bias_flat = add_bias.view(OC)

    # Allocate output
    y = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten number of output positions
    P = N * H_out * W_out

    # Strides
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw = weight.stride()
    stride_y_n, stride_y_oc, stride_y_h, stride_y_w = y.stride()

    BLOCK_M = 32
    BLOCK_N = 32

    grid = lambda META: (
        triton.cdiv(P, META["BLOCK_M"]),
        triton.cdiv(OC, META["BLOCK_N"]),
    )

    conv2d_relu_double_bias_kernel[grid](
        x, weight, conv_bias_flat, add_bias_flat, y,
        N, C_in, H, W,
        OC, H_out, W_out,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_oc, stride_y_h, stride_y_w,
        K_H, K_W,
        P,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-based replacement for:
        Conv2d(in_channels, out_channels, kernel_size, bias=True) -> ReLU -> + extra_bias

    The computation is:
        y = ReLU(conv2d(x, conv.weight) + conv.bias) + bias
    where `bias` is an extra learnable bias broadcast over spatial dimensions.

    Parameter interface matches the original:
        - self.conv.weight : Conv2d weight  [out_channels, in_channels, k_h, k_w]
        - self.conv.bias   : Conv2d bias    [out_channels]
        - self.bias        : extra bias     [*bias_shape], typically (out_channels,) or (out_channels, 1, 1)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size

        # Conv2d module is used as a convenient container for weight & bias parameters
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(k_h, k_w),
            bias=True,
        )
        # Extra bias after ReLU, broadcast over spatial dims
        self.bias = nn.Parameter(
            torch.randn(*bias_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_conv_relu_double_bias(
            x,
            self.conv.weight,
            self.conv.bias,
            self.bias,
        )
