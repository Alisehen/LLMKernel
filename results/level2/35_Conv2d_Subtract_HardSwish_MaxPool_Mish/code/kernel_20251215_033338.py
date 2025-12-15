import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_sub_hswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H_in, W_in,
    C_out, K_H, K_W,
    H_out, W_out,
    subtract_value,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,  # positions (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # output channels
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Flattened spatial+batch dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode flattened index -> (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator for conv output
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Direct convolution loops over C_in, K_H, K_W
    for ic in range(0, C_in):
        for kh in range(0, K_H):
            for kw in range(0, K_W):
                in_h = oh_idx + kh
                in_w = ow_idx + kw

                # Input pointers for this (ic, kh, kw)
                x_ptrs = (
                    x_ptr
                    + n_idx * stride_x_n
                    + ic * stride_x_c
                    + in_h * stride_x_h
                    + in_w * stride_x_w
                )
                x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)

                # Weight pointers for this (ic, kh, kw)
                w_ptrs = (
                    w_ptr
                    + offs_n * stride_w_co
                    + ic * stride_w_ci
                    + kh * stride_w_kh
                    + kw * stride_w_kw
                )
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)

                # FMA: broadcast input over channels
                acc += x_vals[:, None] * w_vals[None, :]

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Subtract scalar value
    acc -= subtract_value

    # HardSwish: x * relu6(x + 3) / 6
    x = acc
    tmp = x + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    acc = x * tmp * (1.0 / 6.0)

    # Store output
    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_out_n
        + offs_n[None, :] * stride_out_c
        + oh_idx[:, None] * stride_out_h
        + ow_idx[:, None] * stride_out_w
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_sub_hswish(x, weight, bias, subtract_value):
    """
    x:        [N, C_in, H_in, W_in]
    weight:   [C_out, C_in, K_H, K_W]
    bias:     [C_out]
    returns:  [N, C_out, H_out, W_out] with
              H_out = H_in - K_H + 1, W_out = W_in - K_W + 1
              and fused: conv2d + bias - subtract_value + HardSwish
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    # Valid conv with stride=1, padding=0, dilation=1 (matching nn.Conv2d default)
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x_contig.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w_contig.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = out.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_sub_hswish_kernel[grid](
        x_contig, w_contig, b_contig, out,
        N, C_in, H_in, W_in,
        C_out, K_H, K_W,
        H_out, W_out,
        subtract_value,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_M=32,
        BLOCK_N=32,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the model:
    Conv2d -> subtract_value -> HardSwish -> MaxPool2d -> Mish

    Fused in Triton:
        Conv2d + bias - subtract_value + HardSwish
    Remaining in PyTorch:
        MaxPool2d, Mish
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Parameters equivalent to nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize like nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.subtract_value = float(subtract_value)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # x expected on CUDA by the harness; Triton kernel requires CUDA tensors
        x = conv2d_sub_hswish(x, self.weight, self.bias, self.subtract_value)
        x = self.pool(x)
        x = F.mish(x)
        return x
