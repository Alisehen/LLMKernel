import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv3d_fwd_kernel(
    x_ptr,          # float32[N, C_in, D_in, H_in, W_in]
    w_ptr,          # float32[C_out, C_in, K_d, K_h, K_w]
    b_ptr,          # float32[C_out] or unused if HAS_BIAS=False
    y_ptr,          # float32[N, C_out, D_out, H_out, W_out]
    N,              # batch size
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    total_positions,  # N * D_out * H_out * W_out
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_p = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)

    # Tile of output spatial positions (flattened)
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < total_positions  # [BLOCK_P]

    # Tile of output channels
    oc_start = pid_oc * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < C_OUT  # [BLOCK_OC]

    # Decode flattened position -> (n, od, oh, ow)
    tmp = p_offsets
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n = tmp // D_out

    # Precompute some constants
    in_HW = H_in * W_in
    in_DHW = D_in * in_HW
    out_HW = H_out * W_out
    out_DHW = D_out * out_HW
    vol_out = out_DHW

    # n_factor_out: [BLOCK_P] base output index for oc=0
    n_factor_out = n * (C_OUT * vol_out)
    spatial_off = od * out_HW + oh * W_out + ow
    base_out = n_factor_out + spatial_off  # [BLOCK_P]

    # n_factor_in: [BLOCK_P] base input index for (cin=0, d=0, h=0, w=0)
    n_factor_in = n * (C_IN * in_DHW)

    # Initialize accumulator for this tile: [BLOCK_OC, BLOCK_P]
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # Strides for weights (flattened (C_out, C_in, K_D, K_H, K_W))
    R = K_D * K_H * K_W
    w_stride_oc = C_IN * R
    w_stride_ci = R
    w_stride_kd = K_H * K_W

    # Main convolution loops: over kernel and input channels (all compile-time)
    for kd in range(K_D):
        id_ = od * stride_d - pad_d + kd * dil_d  # [BLOCK_P]
        valid_d = (id_ >= 0) & (id_ < D_in)
        for kh in range(K_H):
            ih = oh * stride_h - pad_h + kh * dil_h  # [BLOCK_P]
            valid_h = (ih >= 0) & (ih < H_in)
            for kw in range(K_W):
                iw = ow * stride_w - pad_w + kw * dil_w  # [BLOCK_P]
                valid_w = (iw >= 0) & (iw < W_in)
                mask_spatial = mask_p & valid_d & valid_h & valid_w  # [BLOCK_P]

                w_spatial_base = kd * w_stride_kd + kh * K_W + kw

                for ci in range(C_IN):
                    # Input offsets for this (ci, kd, kh, kw) over P positions
                    cin_offset = ci * in_DHW
                    x_offsets = n_factor_in + cin_offset + id_ * in_HW + ih * W_in + iw
                    x_vals = tl.load(
                        x_ptr + x_offsets,
                        mask=mask_spatial,
                        other=0.0,
                    )  # [BLOCK_P]

                    # Weight offsets for this (ci, kd, kh, kw) over OC tile
                    w_offsets = oc_offsets * w_stride_oc + ci * w_stride_ci + w_spatial_base
                    w_vals = tl.load(
                        w_ptr + w_offsets,
                        mask=mask_oc,
                        other=0.0,
                    )  # [BLOCK_OC]

                    x_vals_f32 = tl.cast(x_vals, tl.float32)  # [BLOCK_P]
                    w_vals_f32 = tl.cast(w_vals, tl.float32)  # [BLOCK_OC]

                    # Outer product: [BLOCK_OC, 1] * [1, BLOCK_P] -> [BLOCK_OC, BLOCK_P]
                    prod = w_vals_f32[:, None] * x_vals_f32[None, :]
                    acc += prod

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc_offsets, mask=mask_oc, other=0.0)  # [BLOCK_OC]
        bias_vals_f32 = tl.cast(bias_vals, tl.float32)
        acc += bias_vals_f32[:, None]

    # Store result
    out_offsets = base_out[None, :] + oc_offsets[:, None] * vol_out  # [BLOCK_OC, BLOCK_P]
    mask_out = mask_oc[:, None] & mask_p[None, :]
    tl.store(
        y_ptr + out_offsets,
        acc,
        mask=mask_out,
    )


def triton_conv3d(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None,
                  stride,
                  padding,
                  dilation,
                  groups: int) -> torch.Tensor:
    # Fallback for grouped convs (kernel optimized for groups=1)
    if groups != 1:
        return torch.nn.functional.conv3d(x, weight, bias, stride, padding, dilation, groups)

    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_d, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels mismatch."

    # Normalize stride/padding/dilation to 3-tuples
    if isinstance(stride, int):
        stride_d, stride_h, stride_w = stride, stride, stride
    else:
        stride_d, stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_d, pad_h, pad_w = padding, padding, padding
    else:
        pad_d, pad_h, pad_w = padding
    if isinstance(dilation, int):
        dil_d, dil_h, dil_w = dilation, dilation, dilation
    else:
        dil_d, dil_h, dil_w = dilation

    # Output dimensions (standard conv formula)
    D_out = (D_in + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out),
                    device=x.device,
                    dtype=x.dtype)

    total_positions = N * D_out * H_out * W_out

    BLOCK_P = 64   # tile of output positions
    BLOCK_OC = 32  # tile of output channels

    has_bias = bias is not None
    b_ptr = bias if bias is not None else weight  # dummy pointer if no bias

    grid = lambda meta: (
        triton.cdiv(total_positions, meta['BLOCK_P']),
        triton.cdiv(C_out,          meta['BLOCK_OC']),
    )

    conv3d_fwd_kernel[grid](
        x, weight, b_ptr, y,
        N, D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        total_positions,
        BLOCK_P=BLOCK_P,
        BLOCK_OC=BLOCK_OC,
        C_IN=C_in,
        C_OUT=C_out,
        K_D=K_d,
        K_H=K_h,
        K_W=K_w,
        HAS_BIAS=has_bias,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for a standard 3D convolution.

    Parameters mirror nn.Conv3d; internally keeps an nn.Conv3d module
    for parameter storage/initialization, but the forward pass is
    executed using a custom Triton kernel for high performance.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Use nn.Conv3d for weight/bias management (init, state_dict, etc.)
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv3d.weight
        b = self.conv3d.bias
        return triton_conv3d(
            x,
            w,
            b,
            self.conv3d.stride,
            self.conv3d.padding,
            self.conv3d.dilation,
            self.conv3d.groups,
        )
