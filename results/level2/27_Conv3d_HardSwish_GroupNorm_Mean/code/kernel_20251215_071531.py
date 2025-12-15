import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_hswish_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, C_in, D, H, W,
    OC, Kd, Kh, Kw,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    P,
    BLOCK_M: tl.constexpr,   # tile over flattened (N, D_out, H_out, W_out)
    BLOCK_N: tl.constexpr,   # tile over output channels
    BLOCK_K: tl.constexpr,   # tile over reduction dim C_in * Kd * Kh * Kw
):
    """
    Fused Conv3D (stride=1, padding=0, dilation=1, groups=1) + HardSwish.
    Implemented as a tiled GEMM over:

      M = B * D_out * H_out * W_out        (output positions)
      N = OC                               (output channels)
      K = C_in * Kd * Kh * Kw              (reduction dim: input channels * kernel volume)

    No intermediate stores: all intermediate values live in registers until final store.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program's tile in M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < OC

    # Decode flattened position index offs_m -> (n, od, oh, ow)
    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    od_idx = rem // HW_out
    rem2 = rem % HW_out
    oh_idx = rem2 // W_out
    ow_idx = rem2 % W_out

    # Total reduction size
    K_tot = C_in * Kd * Kh * Kw

    # Accumulator tile [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K dimension in BLOCK_K chunks
    for k_start in range(0, K_tot, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_tot

        # Decode K index -> (ic, kd, kh, kw)
        tmp = offs_k
        kw_idx = tmp % Kw
        tmp = tmp // Kw
        kh_idx = tmp % Kh
        tmp = tmp // Kh
        kd_idx = tmp % Kd
        ic_idx = tmp // Kd

        # Compute input coordinates for this (M, K) tile
        # d_in/h_in/w_in: [BLOCK_M, BLOCK_K]
        d_in = od_idx[:, None] + kd_idx[None, :]
        h_in = oh_idx[:, None] + kh_idx[None, :]
        w_in = ow_idx[:, None] + kw_idx[None, :]
        ic_b = ic_idx[None, :]

        # X pointers: x[n, ic, d_in, h_in, w_in]
        x_offsets = (
            n_idx[:, None] * stride_xn
            + ic_b * stride_xc
            + d_in * stride_xd
            + h_in * stride_xh
            + w_in * stride_xw
        )
        x_ptrs = x_ptr + x_offsets

        # Mask for x tile loads
        mask_x = mask_m[:, None] & mask_k[None, :]

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W pointers: w[oc, ic, kd, kh, kw]
        # We want W tile as [BLOCK_K, BLOCK_N] (K, N) for tl.dot
        oc_b = offs_n[None, :]
        ic_k = ic_idx[:, None]
        kd_k = kd_idx[:, None]
        kh_k = kh_idx[:, None]
        kw_k = kw_idx[:, None]

        w_offsets = (
            oc_b * stride_wo
            + ic_k * stride_wi
            + kd_k * stride_wkd
            + kh_k * stride_wkh
            + kw_k * stride_wkw
        )
        w_ptrs = w_ptr + w_offsets

        mask_w = mask_k[:, None] & mask_n[None, :]

        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate: [M, K] @ [K, N] -> [M, N]
        acc += tl.dot(x_tile, w_tile)

    # Add bias: bias[oc]
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += bias_vals[None, :]

    # HardSwish: x * relu6(x + 3) / 6
    x_add3 = acc + 3.0
    relu6 = tl.minimum(tl.maximum(x_add3, 0.0), 6.0)
    acc = acc * (relu6 * (1.0 / 6.0))

    # Store result: y[n, oc, od, oh, ow]
    y_offsets = (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    y_ptrs = y_ptr + y_offsets
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask)


def conv3d_hswish_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper for conv3d_hswish_kernel.

    x:      [B, C_in, D, H, W]
    weight: [OC, C_in, Kd, Kh, Kw]
    bias:   [OC] or None
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert weight.is_cuda, "Weight must be on CUDA for Triton kernels."
    B, C_in, D, H, W = x.shape
    OC, Ci, Kd, Kh, Kw = weight.shape
    assert Ci == C_in, "in_channels mismatch between input and weight."
    assert Kd > 0 and Kh > 0 and Kw > 0

    # Output dims for stride=1, padding=0, dilation=1
    D_out = D - Kd + 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    assert D_out > 0 and H_out > 0 and W_out > 0, "Invalid Conv3D output size."

    if bias is None:
        bias = x.new_zeros(OC)
    assert bias.shape[0] == OC

    y = torch.empty((B, OC, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flattened spatial+batch dimension
    P = B * D_out * H_out * W_out

    # Strides
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_wo, stride_wi, stride_wkd, stride_wkh, stride_wkw = weight.stride()
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw = y.stride()

    # Kernel launch grid
    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(OC, meta["BLOCK_N"]),
        )

    conv3d_hswish_kernel[grid](
        x, weight, bias, y,
        B, C_in, D, H, W,
        OC, Kd, Kh, Kw,
        D_out, H_out, W_out,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_wo, stride_wi, stride_wkd, stride_wkh, stride_wkw,
        stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
        P,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original model.

    Fuses:
      - Conv3D
      - HardSwish activation
    using a custom Triton kernel, then applies:
      - GroupNorm
      - Mean pooling across spatial dims
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()

        # Conv3D submodule so parameters are `conv.weight` / `conv.bias`
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )

        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # Conv3D + HardSwish via Triton using the conv parameters
        x = conv3d_hswish_triton(x, self.conv.weight, self.conv.bias)
        # GroupNorm (over channels and spatial dims, per sample)
        x = self.group_norm(x)
        # Mean over spatial dimensions -> (B, C)
        x = x.mean(dim=(2, 3, 4))
        return x
