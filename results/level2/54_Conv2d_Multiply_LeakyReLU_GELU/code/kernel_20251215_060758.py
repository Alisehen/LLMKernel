# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline: good for high register pressure / fused ops
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger M tile: better math/launch ratio when P is large
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider N tile + more warps: good when C_OUT large and regs allow
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=['N', 'C_OUT', 'H_out', 'W_out', 'C_IN'],
)
@triton.jit
def conv2d_mul_lrelu_gelu_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, y_ptr,
    N, C_OUT, H_out, W_out, P,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    negative_slope, inv_sqrt2,
    C_IN: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: M = N * H_out * W_out, N-dim = C_OUT
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_OUT

    # Decode flattened spatial index: offs_m -> (n_idx, oh_idx, ow_idx)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Restructured reduction:
    #   Iterate explicitly over (kh, kw, ic_chunk)
    #   -> avoids integer div/mod in the inner loop
    for kh in range(0, K_H):
        x_h = oh_idx[:, None] + kh
        for kw in range(0, K_W):
            x_w = ow_idx[:, None] + kw
            for ic0 in range(0, C_IN, BLOCK_K):
                ic_offsets = ic0 + tl.arange(0, BLOCK_K)
                mask_k = ic_offsets < C_IN

                # Input tile: [BLOCK_M, BLOCK_K]
                x_ptrs = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + ic_offsets[None, :] * stride_xc
                    + x_h * stride_xh
                    + x_w * stride_xw
                )

                # Weight tile: [BLOCK_K, BLOCK_N]
                w_ptrs = (
                    w_ptr
                    + offs_n[None, :] * stride_wo
                    + ic_offsets[:, None] * stride_wc
                    + kh * stride_wh
                    + kw * stride_ww
                )

                x_vals = tl.load(
                    x_ptrs,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                w_vals = tl.load(
                    w_ptrs,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )

                # Tensor-core-friendly dot (TF32 on FP32)
                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # ---- Fused epilogue: bias -> channel-wise mul -> LeakyReLU -> GELU ----

    # Bias add: [C_OUT] broadcast over BLOCK_M
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Channel-wise multiplier: [C_OUT] broadcast
    mult_vals = tl.load(mult_ptr + offs_n, mask=mask_n, other=1.0)
    acc *= mult_vals[None, :]

    # LeakyReLU
    acc = tl.where(acc >= 0.0, acc, acc * negative_slope)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    t = acc * inv_sqrt2
    erf_t = tl.math.erf(t)
    acc = 0.5 * acc * (1.0 + erf_t)

    # Store output
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fused_conv2d_mul_lrelu_gelu(x, weight, bias, multiplier, negative_slope=0.01):
    """
    x: [N, C_in, H_in, W_in] (float32, CUDA)
    weight: [C_out, C_in, K_h, K_w] (float32, CUDA)
    bias: [C_out] (float32, CUDA)
    multiplier: [C_out] or [C_out, 1, 1] (float32, CUDA)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda and bias.is_cuda and multiplier.is_cuda
    assert x.dtype == torch.float32, "Kernel assumes float32 inputs"

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels must match"
    assert K_h == K_w, "Kernel assumes square kernels"

    # Valid convolution, stride=1, padding=0, dilation=1
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1
    assert H_out > 0 and W_out > 0, "Invalid output size"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten multiplier to [C_out]
    if multiplier.dim() == 3:
        mult_flat = multiplier.view(-1)
    else:
        mult_flat = multiplier
    assert mult_flat.numel() == C_out

    P = N * H_out * W_out

    x_strides = x.stride()
    w_strides = weight.stride()
    y_strides = y.stride()

    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    inv_sqrt2 = 1.0 / (2.0 ** 0.5)

    conv2d_mul_lrelu_gelu_kernel[grid](
        x, weight, bias, mult_flat, y,
        N, C_out, H_out, W_out, P,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        w_strides[0], w_strides[1], w_strides[2], w_strides[3],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3],
        negative_slope, inv_sqrt2,
        C_IN=C_in, K_H=K_h, K_W=K_w,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-fused:
      Conv2d (valid, stride=1) -> channel-wise multiply -> LeakyReLU -> GELU
    Implemented as a high-performance GEMM-style convolution on Ada (4090).
    """

    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # Match default nn.LeakyReLU()
        self.negative_slope = 0.01

    def forward(self, x):
        return fused_conv2d_mul_lrelu_gelu(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )
