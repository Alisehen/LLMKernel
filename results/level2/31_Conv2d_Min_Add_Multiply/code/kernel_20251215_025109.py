import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_min_bias_scale_kernel(
    x_ptr,            # [N, C_in, H_in, W_in]
    w_ptr,            # [C_out, C_in, K_H, K_W]
    conv_bias_ptr,    # [C_out]
    extra_bias_ptr,   # [C_out, 1, 1]
    out_ptr,          # [N, C_out, H_out, W_out]
    N,                # batch size
    H_in, W_in,
    C_out,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_cb0,
    stride_eb0,
    stride_on, stride_oc, stride_oh, stride_ow,
    const_val,        # scalar (float32)
    scaling,          # scalar (float32)
    C_in: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    K_TOTAL: tl.constexpr,
    KH_KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 2D matmul-style grid over output:
    #   M axis: P = N * H_out * W_out
    #   N axis: C_out
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)  # along P = N * H_out * W_out
    pid_n = tl.program_id(1)  # along C_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * H_out * W_out
    valid_m = offs_m < P
    valid_n = offs_n < C_out

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Decode offs_m -> (n, oh, ow)
    DHW = H_out * W_out
    n_idx = offs_m // DHW
    rem = offs_m % DHW
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Implicit GEMM over K = C_in * K_H * K_W
    #   X_im2col: [P, K]
    #   W       : [C_out, K]
    #   Out     : [P, C_out]
    # Tiles: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
    # -------------------------------------------------------------------------
    for k0 in tl.static_range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        # Map K index -> (ic, kh, kw)
        ic_idx = offs_k // KH_KW
        remk = offs_k % KH_KW
        kh_idx = remk // K_W
        kw_idx = remk % K_W

        # ---------------------------------------------------------------------
        # Load X tile: [BLOCK_M, BLOCK_K]
        # X[n, ic, oh+kh, ow+kw]
        # ---------------------------------------------------------------------
        ih = oh_idx[:, None] + kh_idx[None, :]
        iw = ow_idx[:, None] + kw_idx[None, :]

        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ic_idx[None, :] * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        x_mask = valid_m[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # ---------------------------------------------------------------------
        # Load W tile: [BLOCK_K, BLOCK_N]
        # W[oc, ic, kh, kw]
        # ---------------------------------------------------------------------
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ic_idx[:, None] * stride_wc
            + kh_idx[:, None] * stride_wkh
            + kw_idx[:, None] * stride_wkw
        )
        w_mask = k_mask[:, None] & valid_n[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # ---------------------------------------------------------------------
        # Matrix multiply accumulate on the tile
        # ---------------------------------------------------------------------
        acc += tl.dot(x_vals, w_vals, out_dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Fused post-ops: conv_bias -> min(const) -> extra_bias -> scale
    # -------------------------------------------------------------------------
    # conv bias: [C_out] -> broadcast over BLOCK_M
    cb_ptrs = conv_bias_ptr + offs_n * stride_cb0
    cb = tl.load(cb_ptrs, mask=valid_n, other=0.0).to(tl.float32)
    acc += cb[None, :]

    # elementwise min with constant
    const_val_f32 = const_val  # already float32 scalar
    acc = tl.where(acc < const_val_f32, acc, const_val_f32)

    # extra bias: [C_out,1,1] with stride along dim0
    eb_ptrs = extra_bias_ptr + offs_n * stride_eb0
    eb = tl.load(eb_ptrs, mask=valid_n, other=0.0).to(tl.float32)
    acc += eb[None, :]

    # scale
    acc = acc * scaling

    # -------------------------------------------------------------------------
    # Store output: [N, C_out, H_out, W_out]
    # -------------------------------------------------------------------------
    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + oh_idx[:, None] * stride_oh
        + ow_idx[:, None] * stride_ow
    )
    store_mask = valid_m[:, None] & valid_n[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def conv2d_min_bias_scale_triton(x, weight, conv_bias, extra_bias, constant_value, scaling_factor):
    """
    x          : [N, C_in, H_in, W_in]
    weight     : [C_out, C_in, K_H, K_W]
    conv_bias  : [C_out]
    extra_bias : [C_out, 1, 1]
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel"
    assert x.ndim == 4 and weight.ndim == 4
    assert conv_bias.ndim == 1
    assert extra_bias.ndim == 3

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in_w == C_in, "Weight C_in mismatch with input"

    # Assume stride=1, padding=0, dilation=1, groups=1
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert H_out > 0 and W_out > 0

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Use existing strides; layout-agnostic
    sx0, sx1, sx2, sx3 = x.stride()
    sw0, sw1, sw2, sw3 = weight.stride()
    scb0 = conv_bias.stride(0)
    seb0 = extra_bias.stride(0)
    so0, so1, so2, so3 = out.stride()

    # Grid over P = N * H_out * W_out and C_out
    P = N * H_out * W_out

    # Tuned tile sizes (can be autotuned for specific GPUs)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(P, BLOCK_M),
        triton.cdiv(C_out, BLOCK_N),
    )

    # Precompute constexpr K-dimension sizes on host
    K_TOTAL = C_in * K_H * K_W
    KH_KW = K_H * K_W

    conv2d_min_bias_scale_kernel[grid](
        x,
        weight,
        conv_bias,
        extra_bias,
        out,
        N,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        sx0,
        sx1,
        sx2,
        sx3,
        sw0,
        sw1,
        sw2,
        sw3,
        scb0,
        seb0,
        so0,
        so1,
        so2,
        so3,
        float(constant_value),
        float(scaling_factor),
        C_in=C_in,
        K_H=K_H,
        K_W=K_W,
        K_TOTAL=K_TOTAL,
        KH_KW=KH_KW,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:

        x = conv2d(x)
        x = min(x, constant_value)
        x = x + bias
        x = x * scaling_factor

    where conv2d has its own bias term.
    """

    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)

        # Conv2d-like parameters (weight + conv bias)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k_h, k_w)
        )
        self.conv_bias = nn.Parameter(
            torch.randn(out_channels)
        )

        # Extra bias added after min, broadcast over spatial
        self.bias = nn.Parameter(torch.randn(*bias_shape))

        # Scalars
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        # x is assumed to be [N, C_in, H, W], typically on CUDA
        return conv2d_min_bias_scale_triton(
            x,
            self.weight,
            self.conv_bias,
            self.bias,
            self.constant_value,
            self.scaling_factor,
        )
