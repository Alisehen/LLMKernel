import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["P", "OC", "K"],
)
@triton.jit
def conv2d_sub_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    subtract1, subtract2,
    N, C_in, H, W,
    OC, KH, KW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    P, K,  # P = N * H_out * W_out, K = C_in * KH * KW
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id -> tiles over output matrix [P, OC]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in [0, P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in [0, OC)

    m_mask = offs_m < P
    n_mask = offs_n < OC

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Decode flattened spatial+batch index -> (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator for conv result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = C_in * KH * KW using matmul-style tiling
    KH_KW = KH * KW

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Decode K index -> (c, kh, kw)
        c_idx = offs_k // KH_KW
        rem_k = offs_k % KH_KW
        kh_idx = rem_k // KW
        kw_idx = rem_k % KW

        # Build input tile A: [BLOCK_M, BLOCK_K]
        # For each (m, k): x[n, c, oh+kh, ow+kw]
        ih = oh_idx[:, None] + kh_idx[None, :]
        iw = ow_idx[:, None] + kw_idx[None, :]

        x_offsets = (
            n_idx[:, None] * stride_xn
            + c_idx[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        x_ptrs = x_ptr + x_offsets

        x_valid = (
            m_mask[:, None] & k_mask[None, :]
            & (ih < H) & (iw < W)
        )

        a = tl.load(x_ptrs, mask=x_valid, other=0.0)

        # Build weight tile B: [BLOCK_K, BLOCK_N]
        # For each (k, n): w[oc, c, kh, kw]
        w_offsets = (
            offs_n[None, :] * stride_wo
            + c_idx[:, None] * stride_wc
            + kh_idx[:, None] * stride_wkh
            + kw_idx[:, None] * stride_wkw
        )
        w_ptrs = w_ptr + w_offsets

        w_valid = k_mask[:, None] & n_mask[None, :]

        b = tl.load(w_ptrs, mask=w_valid, other=0.0)

        # Matmul fragment
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # Fused bias add (if present), scalar subtractions, and Mish activation.
    # All fused ops share the same (offs_m, offs_n, out_mask).
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    sub = subtract1 + subtract2
    acc = acc - sub

    # Mish: x * tanh(softplus(x)), with softplus(x) = log(1 + exp(x))
    # Implement tanh via exponentials to respect "no tl.tanh" constraint.
    softplus = tl.log(1.0 + tl.exp(acc))
    t = tl.exp(2.0 * softplus)
    tanh_sp = (t - 1.0) / (t + 1.0)
    out_vals = acc * tanh_sp

    # Store to out[N, OC, H_out, W_out] (contiguous NCHW layout)
    out_offsets = (
        n_idx[:, None] * (OC * H_out * W_out)
        + offs_n[None, :] * (H_out * W_out)
        + oh_idx[:, None] * W_out
        + ow_idx[:, None]
    )
    out_ptrs = out_ptr + out_offsets
    out_mask = m_mask[:, None] & n_mask[None, :]

    tl.store(out_ptrs, out_vals, mask=out_mask)


def conv2d_sub_sub_mish(x, weight, bias, subtract1: float, subtract2: float):
    """
    x:      [N, C_in, H, W] CUDA tensor
    weight: [OC, C_in, KH, KW]
    bias:   [OC] or None
    Computes:
      y = mish(conv2d(x, weight, bias) - subtract1 - subtract2)
    with conv2d: stride=1, padding=0, dilation=1.
    """
    assert x.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda, "Weight must be on CUDA"
    assert x.ndim == 4 and weight.ndim == 4

    N, C_in, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert Cw == C_in, "Weight in_channels must match input"

    # No padding / stride / dilation supported here
    H_out = H - KH + 1
    W_out = W - KW + 1
    assert H_out > 0 and W_out > 0, "Invalid H/W for given kernel size (no padding assumed)"

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    has_bias = bias is not None
    b_contig = bias.contiguous() if has_bias else None

    # Output in float32 (accumulator precision), cast back at the end if needed
    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=torch.float32)

    # Strides in elements
    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_wo, stride_wc, stride_wkh, stride_wkw = w_contig.stride()

    P = N * H_out * W_out
    K = C_in * KH * KW

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(OC, meta["BLOCK_N"]),
        )

    # Bias pointer: if absent, pass a valid tensor but guard with HAS_BIAS
    b_arg = b_contig if has_bias else x_contig

    conv2d_sub_sub_mish_kernel[grid](
        x_contig, w_contig, b_arg, out,
        float(subtract1), float(subtract2),
        N, C_in, H, W,
        OC, KH, KW, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wkh, stride_wkw,
        P, K,
        HAS_BIAS=has_bias,
    )

    if out.dtype != x.dtype:
        out = out.to(x.dtype)
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated:
      y = mish(conv2d(x, weight, bias) - subtract_value_1 - subtract_value_2)
    with conv2d: stride=1, padding=0, dilation=1, NCHW layout.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
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
