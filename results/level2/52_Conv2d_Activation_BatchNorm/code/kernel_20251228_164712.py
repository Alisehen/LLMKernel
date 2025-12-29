import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_softplus_tanh_mul_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, Kh, Kw,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over N*H_out*W_out
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # reduction tile over C_in*Kh*Kw
):
    pid_m = tl.program_id(0)  # over output spatial+batch (M dimension)
    pid_n = tl.program_id(1)  # over output channels (N dimension)

    M = N * H_out * W_out
    K_tot = C_in * Kh * Kw

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, h_out, w_out)
    hw = H_out * W_out
    n_idx = offs_m // hw
    rem_hw = offs_m % hw
    h_out_idx = rem_hw // W_out
    w_out_idx = rem_hw % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_tot, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_tot

        # Decode reduction index k -> (c_in, kh, kw)
        khw = Kh * Kw
        ci = offs_k // khw
        rem_k = offs_k % khw
        kh_idx = rem_k // Kw
        kw_idx = rem_k % Kw

        # Input coordinates
        h_in = h_out_idx[:, None] + kh_idx[None, :]
        w_in = w_out_idx[:, None] + kw_idx[None, :]

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ci[None, :] * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        a = tl.load(
            x_ptrs,
            mask=(mask_m[:, None] & mask_k[None, :]),
            other=0.0,
        )

        # Weight coordinates: (C_out, C_in, Kh, Kw)
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ci[:, None] * stride_wc
            + kh_idx[:, None] * stride_wh
            + kw_idx[:, None] * stride_ww
        )
        b = tl.load(
            w_ptrs,
            mask=(mask_k[:, None] & mask_n[None, :]),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Fused activation: x * tanh(softplus(x))
    # softplus(x) ≈ log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    # tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
    t2 = tl.exp(2.0 * sp)
    t = (t2 - 1.0) / (t2 + 1.0)
    acc = t * acc

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + h_out_idx[:, None] * stride_yh
        + w_out_idx[:, None] * stride_yw
    )

    tl.store(
        y_ptrs,
        acc,
        mask=(mask_m[:, None] & mask_n[None, :]),
    )


def conv_softplus_tanh_mul_triton(x, weight, bias):
    """
    Fused:
      y = conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)
      y = y * tanh(softplus(y))
    """
    assert x.dim() == 4, "Input must be NCHW"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, Kh, Kw = weight.shape
    assert C_in_w == C_in, "in_channels mismatch between input and weight"
    # Assumes stride=1, padding=0, dilation=1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv_softplus_tanh_mul_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, Kh, Kw,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,  # power-of-2 as required
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement:
      - Conv2d implemented via fused Triton kernel + activation
      - BatchNorm2d kept as PyTorch op (can be fused separately if desired)
    """

    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep a Conv2d module only for parameters / state_dict compatibility
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = conv_softplus_tanh_mul_triton(x, self.conv.weight, self.conv.bias)
        x = self.bn(x)
        return x
