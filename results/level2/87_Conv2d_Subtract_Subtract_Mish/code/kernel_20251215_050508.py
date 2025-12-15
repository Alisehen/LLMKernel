import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_sub_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    subtract1, subtract2,
    N, C_in, H, W,
    OC, KH, KW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Flatten output [N, OC, H_out, W_out] into:
    #   P = N * H_out * W_out  (rows, M dimension)
    #   OC                      (cols, N dimension)
    P = N * H_out * W_out

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] over P
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N] over OC

    m_valid = offs_m < P
    n_valid = offs_n < OC

    # Decode flattened spatial+batch index into (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator for conv result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution: sum over input channels and kernel spatial positions
    for c in range(0, C_in):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # Input spatial positions
                ih = oh_idx + kh  # [BLOCK_M]
                iw = ow_idx + kw  # [BLOCK_M]

                # Compute input pointer offsets for this (c, kh, kw)
                x_offsets = (
                    n_idx * stride_xn
                    + c * stride_xc
                    + ih * stride_xh
                    + iw * stride_xw
                )
                x_ptrs = x_ptr + x_offsets

                # Mask: only valid for in-bounds and valid m indices
                x_mask = m_valid & (ih < H) & (iw < W)

                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M]

                # Weight offsets for this (c, kh, kw) over oc dimension
                w_offsets = (
                    offs_n * stride_wo
                    + c * stride_wc
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_ptrs = w_ptr + w_offsets

                w_vals = tl.load(w_ptrs, mask=n_valid, other=0.0)  # [BLOCK_N]

                # Outer product accumulate
                acc += x_vals[:, None] * w_vals[None, :]

    # Add bias if present (compile-time flag to avoid pointer-int comparisons)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_valid, other=0.0)  # [BLOCK_N]
        acc += bias[None, :]

    # Subtractions
    acc = acc - subtract1
    acc = acc - subtract2

    # Mish activation: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(acc))
    t = tl.exp(2.0 * softplus)
    tanh_sp = (t - 1.0) / (t + 1.0)
    out_vals = acc * tanh_sp

    # Store result to out[N, OC, H_out, W_out]
    # out layout: NCHW with strides derived in wrapper
    out_offsets = (
        n_idx[:, None] * (OC * H_out * W_out)
        + offs_n[None, :] * (H_out * W_out)
        + oh_idx[:, None] * W_out
        + ow_idx[:, None]
    )

    out_ptrs = out_ptr + out_offsets
    out_mask = m_valid[:, None] & n_valid[None, :]

    tl.store(out_ptrs, out_vals, mask=out_mask)


def conv2d_sub_sub_mish(x, weight, bias, subtract1: float, subtract2: float):
    """
    x:      [N, C_in, H, W] (CUDA tensor)
    weight: [OC, C_in, KH, KW]
    bias:   [OC] or None
    Performs conv2d (stride=1, padding=0, dilation=1), then
      y = conv(x, weight, bias) - subtract1 - subtract2
      y = mish(y)
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert x.ndim == 4
    assert weight.ndim == 4

    N, C_in, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert Cw == C_in, "Weight in_channels must match input"

    # Only support standard conv2d with stride=1, padding=0, dilation=1
    H_out = H - KH + 1
    W_out = W - KW + 1
    assert H_out > 0 and W_out > 0, "Invalid H/W for given kernel size (no padding assumed)"

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    has_bias = bias is not None
    b_contig = bias.contiguous() if has_bias else None

    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=torch.float32)

    # Strides (PyTorch gives strides in element units)
    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_wo, stride_wc, stride_wkh, stride_wkw = w_contig.stride()

    # Flattened spatial dimension for grid
    P = N * H_out * W_out

    BLOCK_M = 64
    BLOCK_N = 64

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = (cdiv(P, BLOCK_M), cdiv(OC, BLOCK_N))

    # When bias is absent, we still pass a valid tensor pointer but guard its use
    # with the compile-time HAS_BIAS flag.
    b_arg = b_contig if has_bias else x_contig

    conv2d_sub_sub_mish_kernel[grid](
        x_contig, w_contig, b_arg, out,
        float(subtract1), float(subtract2),
        N, C_in, H, W,
        OC, KH, KW, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wkh, stride_wkw,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HAS_BIAS=has_bias,
    )

    # Cast back to original dtype if needed
    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version:
      y = mish(conv2d(x) - subtract_value_1 - subtract_value_2)
    with conv2d: stride=1, padding=0, dilation=1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        # Match nn.Conv2d parameter shapes (no padding/stride handling here)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.subtract_value_1 = float(subtract_value_1)
        self.subtract_value_2 = float(subtract_value_2)

    def forward(self, x):
        return conv2d_sub_sub_mish(
            x,
            self.weight,
            self.bias,
            self.subtract_value_1,
            self.subtract_value_2,
        )
