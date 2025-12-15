import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_ncdhw_kernel(
    x_ptr,  # [N, C_in, D_in, H_in, W_in]
    w_ptr,  # [C_out, C_in, K_D, K_H, K_W]
    b_ptr,  # [C_out]
    y_ptr,  # [N, C_out, D_out, H_out, W_out]
    N, C_in, D_in, H_in, W_in,
    C_out, K_D, K_H, K_W,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # block over (N * D_out * H_out * W_out)
    BLOCK_N: tl.constexpr,  # block over C_out
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * D_out * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, od, oh, ow)
    DHW = D_out * H_out * W_out
    HW = H_out * W_out

    n = offs_m // DHW
    rem = offs_m % DHW
    od = rem // HW
    rem2 = rem % HW
    oh = rem2 // W_out
    ow = rem2 % W_out

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution sum over input channels and kernel volume
    for ic in range(0, C_in):
        for kd in range(0, K_D):
            id_in = od + kd
            for kh in range(0, K_H):
                ih_in = oh + kh
                for kw in range(0, K_W):
                    iw_in = ow + kw

                    # Input pointers for this ic, kd, kh, kw, over all offs_m
                    x_ptrs = (
                        x_ptr
                        + n * stride_xn
                        + ic * stride_xc
                        + id_in * stride_xd
                        + ih_in * stride_xh
                        + iw_in * stride_xw
                    )
                    # Weight pointers for this ic, kd, kh, kw, over all offs_n
                    w_ptrs = (
                        w_ptr
                        + offs_n * stride_wn
                        + ic * stride_wc
                        + kd * stride_wd
                        + kh * stride_wh
                        + kw * stride_ww
                    )

                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)

                    x_vals_f32 = x_vals.to(tl.float32)
                    w_vals_f32 = w_vals.to(tl.float32)

                    acc += x_vals_f32[:, None] * w_vals_f32[None, :]

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    bias_vals_f32 = bias_vals.to(tl.float32)
    acc += bias_vals_f32[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od[:, None] * stride_yd
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    mask_store = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


def conv3d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Simple NCDHW Conv3d (stride=1, padding=0, dilation=1) using Triton.

    Args:
        x:      [N, C_in, D_in, H_in, W_in]
        weight: [C_out, C_in, K_D, K_H, K_W]
        bias:   [C_out]

    Returns:
        y: [N, C_out, D_out, H_out, W_out]
           where D_out = D_in - K_D + 1, etc.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in == C_in_w

    D_out = D_in - K_D + 1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert D_out > 0 and H_out > 0 and W_out > 0

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_M = 32  # power-of-2
    BLOCK_N = 32  # power-of-2

    grid = lambda META: (
        triton.cdiv(N * D_out * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv3d_ncdhw_kernel[grid](
        x,
        weight,
        bias,
        y,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        K_D,
        K_H,
        K_W,
        D_out,
        H_out,
        W_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        weight.stride(4),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        y.stride(4),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution with a Triton kernel, applies Group Normalization,
    then computes the mean over (C, D, H, W) to produce shape (N,).
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_d = k_h = k_w = kernel_size
        else:
            k_d, k_h, k_w = kernel_size
        assert k_d == k_h == k_w, "This implementation assumes cubic kernels."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k_d

        # Conv3d parameters (NCDHW)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k_d, k_h, k_w)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize similar to nn.Conv3d default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * k_d * k_h * k_w
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # GroupNorm (operates on [N, C_out, D, H, W])
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton Conv3d
        x = conv3d_triton(x, self.weight, self.bias)
        # GroupNorm + mean over (C, D, H, W)
        x = self.group_norm(x)
        x = x.mean(dim=[1, 2, 3, 4])
        return x.unsqueeze(1)
