# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_gelu_gap_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, C_in, H, W,
    C_out, KH, KW,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yn, stride_yc,
    BLOCK_CO: tl.constexpr,  # block of output channels
    BLOCK_K: tl.constexpr,   # block of K = C_in*KH*KW
    BLOCK_HW: tl.constexpr,  # block of spatial positions
):
    # program ids: one batch element per pid_b, one channel tile per pid_co
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)

    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_out

    # total spatial positions
    HW_out = H_out * W_out

    # accumulator for global sum over spatial positions, per output channel
    acc_sum = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # load bias once per program
    bias = tl.load(bias_ptr + offs_co, mask=mask_co, other=0.0)
    bias = bias.to(tl.float32)

    # loop over spatial positions in blocks
    for hw_start in range(0, HW_out, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW_out

        # decode linear spatial index to (oy, ox)
        oy = offs_hw // W_out
        ox = offs_hw % W_out

        # conv accumulators for this spatial block: [BLOCK_HW, BLOCK_CO]
        out_block = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

        # loop over K = C_in * KH * KW in tiles
        for k_start in range(0, C_in * KH * KW, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < (C_in * KH * KW)

            # decode K index -> (ic, ky, kx)
            ic = offs_k // (KH * KW)
            rem = offs_k % (KH * KW)
            ky = rem // KW
            kx = rem % KW

            # ---- load weight tile [BLOCK_K, BLOCK_CO] ----
            w_ptrs = (
                w_ptr
                + offs_co[None, :] * stride_wo
                + ic[:, None] * stride_wi
                + ky[:, None] * stride_wkh
                + kx[:, None] * stride_wkw
            )
            w_tile = tl.load(
                w_ptrs,
                mask=mask_k[:, None] & mask_co[None, :],
                other=0.0,
            )
            w_tile = w_tile.to(tl.float32)

            # ---- load input tile [BLOCK_HW, BLOCK_K] ----
            in_y = oy[:, None] + ky[None, :]
            in_x = ox[:, None] + kx[None, :]

            x_ptrs = (
                x_ptr
                + pid_b * stride_xn
                + ic[None, :] * stride_xc
                + in_y * stride_xh
                + in_x * stride_xw
            )
            x_tile = tl.load(
                x_ptrs,
                mask=mask_hw[:, None] & mask_k[None, :],
                other=0.0,
            )
            x_tile = x_tile.to(tl.float32)

            # matmul: [BLOCK_HW, BLOCK_K] @ [BLOCK_K, BLOCK_CO]
            out_block += tl.dot(x_tile, w_tile, allow_tf32=True)

        # add bias and apply GELU to out_block
        out_block = out_block + bias[None, :]

        x_fp32 = out_block
        x_cubed = x_fp32 * x_fp32 * x_fp32
        k0 = 0.7978845608028654  # sqrt(2/pi)
        inner = k0 * (x_fp32 + 0.044715 * x_cubed)
        two_inner = 2.0 * inner
        exp2 = tl.exp(two_inner)
        tanh_inner = (exp2 - 1.0) / (exp2 + 1.0)
        out_block = 0.5 * x_fp32 * (1.0 + tanh_inner)

        # mask out invalid spatial rows before reduction
        mask_hw_b = mask_hw[:, None]
        out_block = tl.where(mask_hw_b, out_block, 0.0)

        # reduce over spatial block and accumulate into global sum
        sum_hw = tl.sum(out_block, axis=0)  # [BLOCK_CO]
        acc_sum += sum_hw

    # finalize global average pooling
    denom = tl.float32(HW_out)
    avg = acc_sum / denom

    # store result [B, C_out]
    y_ptrs = y_ptr + pid_b * stride_yn + offs_co * stride_yc
    tl.store(y_ptrs, avg.to(y_ptr.dtype.element_ty), mask=mask_co)


def conv2d_gelu_gap_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Fused implementation of:
      y = Conv2d(x, weight, bias)
      y = GELU(y)
      y = global_avg_pool2d(y)
    Returns tensor of shape (B, C_out).
    """
    B, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels must match in weight"

    H_out = H - KH + 1
    W_out = W - KW + 1

    # output (B, C_out)
    y = torch.empty((B, C_out), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            max(1, B),
            max(1, triton.cdiv(C_out, meta["BLOCK_CO"])),
        )

    conv2d_gelu_gap_kernel[grid](
        x, weight, bias, y,
        B, C_in, H, W,
        C_out, KH, KW,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1),
        BLOCK_CO=64,
        BLOCK_K=64,
        BLOCK_HW=16,
    )
    return y


class ModelNew(nn.Module):
    """
    Equivalent to:

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        forward(x):
            x = self.conv(x)
            x = GELU(x)
            x = global average pooling over H_out x W_out
            return x.view(B, C_out)

    but implemented with a single fused Triton kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return conv2d_gelu_gap_triton(x, self.weight, self.bias)
