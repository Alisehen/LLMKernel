import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # High arithmetic intensity, but reduced pipeline depth for better occupancy
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Narrower N tile to cut register usage (more active warps)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Taller M tile for large spatial sizes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider N tile, small M – good when OC is large, spatial small
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Smaller K tile to reduce register pressure on very deep channels
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            num_warps=4,
            num_stages=2,
        ),
        # Very small tile for extreme register pressure / tiny problems
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=["P", "OC", "K"],
)
@triton.jit
def conv2d_sub_sub_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    subtract,                       # scalar = subtract1 + subtract2 (fused)
    N, C_in, H, W,
    OC, KH, KW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    P, K,                           # P = N * H_out * W_out, K = C_in * KH * KW
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program IDs -> tiles over output matrix [P, OC]
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)  # along P (N * H_out * W_out)
    pid_n = tl.program_id(1)  # along OC

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < P
    n_mask = offs_n < OC

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Decode flattened P-index -> (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m - n_idx * HW_out
    oh_idx = rem // W_out
    ow_idx = rem - oh_idx * W_out

    # -------------------------------------------------------------------------
    # Precompute per-row base pointer offsets for x and out
    # These are invariant across K-loop and N-loop, kept in registers
    # -------------------------------------------------------------------------
    # x_base[m] = (n * stride_xn + oh * stride_xh + ow * stride_xw)
    x_base = (
        n_idx * stride_xn
        + oh_idx * stride_xh
        + ow_idx * stride_xw
    )[:, None]

    # out_base[m] = (n * OC * H_out * W_out + oh * W_out + ow)
    OC_HW_out = OC * H_out * W_out
    out_base = (
        n_idx * OC_HW_out
        + oh_idx * W_out
        + ow_idx
    )[:, None]

    # Precompute per-column base pointer offsets for weights
    # w_oc_base[n] = oc * stride_wo
    w_oc_base = (offs_n * stride_wo)[None, :]

    # -------------------------------------------------------------------------
    # Accumulator for conv result in fp32
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KH_KW = KH * KW

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        tl.multiple_of(offs_k, BLOCK_K)

        # Decode reduction index K -> (c, kh, kw)
        c_idx = offs_k // KH_KW
        rem_k = offs_k - c_idx * KH_KW
        kh_idx = rem_k // KW
        kw_idx = rem_k - kh_idx * KW

        # ---------------------------------------------------------------------
        # Input tile A: [BLOCK_M, BLOCK_K]  x[n, c, oh+kh, ow+kw]
        # ---------------------------------------------------------------------
        # ih = oh + kh, iw = ow + kw
        ih = oh_idx[:, None] + kh_idx[None, :]
        iw = ow_idx[:, None] + kw_idx[None, :]

        x_offsets = (
            x_base
            + c_idx[None, :] * stride_xc
            + kh_idx[None, :] * stride_xh
            + kw_idx[None, :] * stride_xw
        )
        x_ptrs = x_ptr + x_offsets

        x_valid = (
            m_mask[:, None] & k_mask[None, :]
            & (ih < H) & (iw < W)
        )

        a = tl.load(x_ptrs, mask=x_valid, other=0.0)

        # ---------------------------------------------------------------------
        # Weight tile B: [BLOCK_K, BLOCK_N]  w[oc, c, kh, kw]
        # ---------------------------------------------------------------------
        w_offsets = (
            w_oc_base
            + c_idx[:, None] * stride_wc
            + kh_idx[:, None] * stride_wkh
            + kw_idx[:, None] * stride_wkw
        )
        w_ptrs = w_ptr + w_offsets

        w_valid = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=w_valid, other=0.0)

        # FMA accumulate
        acc += tl.dot(a, b)

        k0 += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused: bias (optional), scalar subtraction, Mish activation
    # Mish(x) = x * tanh(softplus(x)), softplus(x)=log(1+exp(x))
    # Here we use: tanh(softplus(x)) = ((1+e)^2 - 1)/((1+e)^2 + 1)
    # requiring a single exp + cheap arithmetic.
    # All kept in registers; NO intermediate stores.
    # -------------------------------------------------------------------------
    if HAS_BIAS:
        # Bias is 1D over OC
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    acc -= subtract  # fused scalar subtraction

    # Mish approximation using one exp
    p = 1.0 + tl.exp(acc)   # 1 + exp(x)
    t = p * p               # (1 + exp(x))^2
    out_vals = acc * (t - 1.0) / (t + 1.0)

    # -------------------------------------------------------------------------
    # Single final store to output tensor [N, OC, H_out, W_out] (NCHW)
    # No intermediate global stores anywhere above.
    # -------------------------------------------------------------------------
    out_offsets = (
        out_base
        + offs_n[None, :] * (H_out * W_out)
    )
    out_ptrs = out_ptr + out_offsets
    out_mask = m_mask[:, None] & n_mask[None, :]

    tl.store(out_ptrs, out_vals, mask=out_mask)


def conv2d_sub_sub_mish(x, weight, bias, subtract1: float, subtract2: float):
    """
    x:      [N, C_in, H, W]  (CUDA tensor, any floating type supported by Triton)
    weight: [OC, C_in, KH, KW]
    bias:   [OC] or None

    Computes:
        y = mish(conv2d(x, weight, bias) - subtract1 - subtract2)

    conv2d: stride=1, padding=0, dilation=1, NCHW layout.
    Accumulation is always in float32 for numerical stability; output is then
    cast back to x.dtype.
    """
    assert x.is_cuda and weight.is_cuda, "Input and weight must be on CUDA"
    assert x.ndim == 4 and weight.ndim == 4

    N, C_in, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert Cw == C_in, "Weight in_channels must match input"

    # No padding / stride / dilation
    H_out = H - KH + 1
    W_out = W - KW + 1
    assert H_out > 0 and W_out > 0, "Invalid H/W for given kernel size (no padding assumed)"

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    has_bias = bias is not None
    b_contig = bias.contiguous() if has_bias else None

    # Output in float32 (accumulator precision)
    out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=torch.float32)

    # Strides in elements (NCHW and OIHW)
    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_wo, stride_wc, stride_wkh, stride_wkw = w_contig.stride()

    P = N * H_out * W_out
    K = C_in * KH * KW

    # Fuse scalar subtractions to reduce register usage in kernel
    subtract = float(subtract1 + subtract2)

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(OC, meta["BLOCK_N"]),
        )

    # Bias pointer: if absent, pass a valid tensor but guard with HAS_BIAS
    b_arg = b_contig if has_bias else x_contig

    conv2d_sub_sub_mish_kernel[grid](
        x_contig, w_contig, b_arg, out,
        subtract,
        N, C_in, H, W,
        OC, KH, KW, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wkh, stride_wkw,
        P, K,
        HAS_BIAS=has_bias,
    )

    # Cast back to input dtype if needed (e.g., fp16/bf16 pipeline)
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
