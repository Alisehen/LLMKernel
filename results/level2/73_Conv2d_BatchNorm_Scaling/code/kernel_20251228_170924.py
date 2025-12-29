import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_bn_scale_kernel(
    x_ptr, w_ptr,
    scale_ptr, shift_ptr,
    y_ptr,
    N, C_in, H_in, W_in,
    C_out, K_h, K_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # rows: N * H_out * W_out
    BLOCK_N: tl.constexpr,  # columns: C_out
    BLOCK_K: tl.constexpr,  # unused but kept for compatibility
):
    # Program IDs for tiles along M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in the flattened output "matrix":
    # M dimension corresponds to (n, h_out, w_out)
    # N dimension corresponds to output channels (C_out)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * H_out * W_out
    HW_out = H_out * W_out

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Map M-index to (n, h_out, w_out)
    n = (offs_m // HW_out)[:, None]          # (BLOCK_M, 1)
    rem = (offs_m % HW_out)[:, None]         # (BLOCK_M, 1)
    h_out = (rem // W_out)                   # (BLOCK_M, 1)
    w_out = (rem % W_out)                    # (BLOCK_M, 1)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Prepare output-channel indices (for weights and scale/shift)
    co = offs_n[None, :]                     # (1, BLOCK_N)

    # Convolution: explicit reduction over C_in, K_h, K_w
    # Each iteration accumulates outer product: x_vals (BM,1) * w_vals (1,BN)
    for c in range(0, C_in):
        # Base pointers that do not depend on kh/kw
        x_base = x_ptr + n * stride_xn + c * stride_xc  # (BM,1) base for this input channel
        w_base = w_ptr + co * stride_wn + c * stride_wc # (1,BN) base for this input channel

        for kh in range(0, K_h):
            h_in = h_out + kh
            h_ok = (h_in >= 0) & (h_in < H_in)

            for kw in range(0, K_w):
                w_in = w_out + kw
                w_ok = (w_in >= 0) & (w_in < W_in)

                # Load input patch values for this (c,kh,kw)
                x_ptrs = x_base + h_in * stride_xh + w_in * stride_xw  # (BM,1)
                mask_x = mask_m[:, None] & h_ok & w_ok
                x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)  # (BM,1)

                # Load weights for this (c,kh,kw) across output channels
                w_ptrs = w_base + kh * stride_wh + kw * stride_ww       # (1,BN)
                mask_w = mask_n[None, :]
                w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)  # (1,BN)

                # Outer-product accumulate
                acc += x_vals * w_vals

    # Apply fused per-channel scale & shift (includes BatchNorm + global scaling)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0)  # (BLOCK_N,)
    shift = tl.load(shift_ptr + offs_n, mask=mask_n, other=0.0)  # (BLOCK_N,)

    acc = acc * scale[None, :] + shift[None, :]

    # Map back to output tensor indices and store
    n_b = n           # (BLOCK_M, 1)
    h_b = h_out
    w_b = w_out
    co_b = co         # (1, BLOCK_N)

    y_ptrs = (
        y_ptr
        + n_b * stride_yn
        + co_b * stride_yc
        + h_b * stride_yh
        + w_b * stride_yw
    )

    mask_y = (
        mask_m[:, None]
        & mask_n[None, :]
        & (h_b >= 0)
        & (h_b < H_out)
        & (w_b >= 0)
        & (w_b < W_out)
    )

    tl.store(y_ptrs, acc, mask=mask_y)


def fused_conv_bn_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    scaling_factor: float,
):
    # x: [N, C_in, H_in, W_in]
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_w_in, K_h, K_w = weight.shape
    assert C_w_in == C_in, "Conv weight in_channels mismatch"

    # Valid convolution: no padding, stride=1, dilation=1
    H_out = H_in - K_h + 1
    W_out = W_in - K_w + 1

    # Compute fused per-channel scale and shift for:
    # y = scaling_factor * BN(conv(x))
    # BN(z) = bn_weight * (z - running_mean) / sqrt(running_var + eps) + bn_bias
    # where z = conv(x) + bias
    inv_std = torch.rsqrt(running_var + eps)
    bn_scale = bn_weight * inv_std  # per-channel

    if bias is not None:
        shift_term = bn_scale * (bias - running_mean)
    else:
        shift_term = bn_scale * (-running_mean)

    scale = scaling_factor * bn_scale
    shift = scaling_factor * (shift_term + bn_bias)

    scale = scale.contiguous()
    shift = shift.contiguous()

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_bn_scale_kernel[grid](
        x, weight,
        scale, shift,
        y,
        N, C_in, H_in, W_in,
        C_out, K_h, K_w,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,   # kept for signature compatibility (unused in kernel)
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized model:
    Fuses Conv2d + BatchNorm2d (inference-style) + scaling into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size

        # Conv parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k_h, k_w)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

        # BatchNorm-like parameters (inference behavior)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.bn_eps = 1e-5

        # Final scaling factor
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return fused_conv_bn_scale(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.bn_eps,
            self.scaling_factor,
        )
