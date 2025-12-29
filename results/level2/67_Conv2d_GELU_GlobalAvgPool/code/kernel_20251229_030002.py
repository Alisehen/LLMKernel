import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_gap_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, C_in, H, W, C_out, K, H_out, W_out, P,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_outb, stride_outc,
    inv_P,
    BLOCK_P: tl.constexpr,  # block over spatial output positions
    BLOCK_R: tl.constexpr,  # block over C_in * K * K reduction
):
    pid = tl.program_id(0)  # one program per (batch, out_channel)
    b = pid // C_out
    oc = pid % C_out

    # Guard against out-of-range pids (shouldn't happen with our grid, but safe)
    if pid >= B * C_out:
        return

    # Base pointer for weights of this output channel
    w_oc_ptr = w_ptr + oc * stride_wco

    # Bias for this output channel
    bias_val = tl.load(bias_ptr + oc)

    # Total reduction length over input channels and kernel window
    R = C_in * K * K

    # Accumulator over spatial positions (for global average pooling)
    acc_sum = tl.zeros((), dtype=tl.float32)

    # Loop over spatial positions in blocks of BLOCK_P
    for p_start in range(0, P, BLOCK_P):
        offs_p = p_start + tl.arange(0, BLOCK_P)
        mask_p = offs_p < P

        # Map flattened spatial index -> (oh, ow)
        oh = offs_p // W_out
        ow = offs_p - oh * W_out

        # Per-position accumulator for convolution result
        val_p = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # Loop over combined (C_in, K, K) reduction dimension in tiles of BLOCK_R
        for r_start in range(0, R, BLOCK_R):
            offs_r = r_start + tl.arange(0, BLOCK_R)
            mask_r = offs_r < R

            # Decode offs_r -> (ic, kh, kw)
            kk = K * K
            ic = offs_r // kk
            rem = offs_r - ic * kk
            kh = rem // K
            kw = rem - kh * K

            # Compute input coordinates for each (p, r) pair
            in_h = oh[:, None] + kh[None, :]
            in_w = ow[:, None] + kw[None, :]

            # Pointers for input x[b, ic, in_h, in_w]
            x_ptrs = (
                x_ptr
                + b * stride_xb
                + ic[None, :] * stride_xc
                + in_h * stride_xh
                + in_w * stride_xw
            )

            mask_x = mask_p[:, None] & mask_r[None, :]

            x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Pointers for weights w[oc, ic, kh, kw]
            w_ptrs = w_oc_ptr + ic * stride_wci + kh * stride_wkh + kw * stride_wkw
            w_vals = tl.load(w_ptrs, mask=mask_r, other=0.0)

            # Accumulate dot-product over BLOCK_R for each position in BLOCK_P
            val_p += tl.sum(x_vals * w_vals[None, :], axis=1)

        # Add bias and apply GELU (tanh-based approximation)
        x_val = val_p + bias_val
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x3 = x_val * x_val * x_val
        inner = c0 * (x_val + c1 * x3)
        t = tl.exp(2.0 * inner)
        tanh_inner = (t - 1.0) / (t + 1.0)
        gelu_val = 0.5 * x_val * (1.0 + tanh_inner)

        # Zero out inactive positions
        gelu_val = tl.where(mask_p, gelu_val, 0.0)

        # Accumulate for global average pooling
        acc_sum += tl.sum(gelu_val, axis=0)

    # Finish global average pooling
    out_val = acc_sum * inv_P

    # Store result: out[b, oc]
    out_ptrs = out_ptr + b * stride_outb + oc * stride_outc
    tl.store(out_ptrs, out_val)


def fused_conv_gelu_gap(x, weight, bias):
    """
    x: (B, C_in, H, W)
    weight: (C_out, C_in, K, K)
    bias: (C_out,)
    returns: (B, C_out) after Conv2d -> GELU -> global average pooling
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel"
    B, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "in_channels mismatch"
    assert K_h == K_w, "Only square kernels supported"
    K = K_h

    H_out = H - K + 1
    W_out = W - K + 1
    P = H_out * W_out
    inv_P = 1.0 / float(P)

    out = torch.empty((B, C_out), device=x.device, dtype=x.dtype)

    N = B * C_out
    grid = lambda META: (triton.cdiv(N, 1),)

    conv_gelu_gap_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W, C_out, K, H_out, W_out, P,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1),
        inv_P,
        BLOCK_P=64,
        BLOCK_R=64,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      Conv2d -> GELU -> global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels supported"
            k = kernel_size[0]
        else:
            k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return fused_conv_gelu_gap(x, self.weight, self.bias)
