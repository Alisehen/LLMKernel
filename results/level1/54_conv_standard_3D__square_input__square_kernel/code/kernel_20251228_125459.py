import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv3d_fwd_gemm_kernel(
    x_ptr,          # float32[N, C_in, D_in, H_in, W_in]
    w_ptr,          # float32[C_out, C_in, K_d, K_h, K_w] (treated as [C_out, K_TOTAL])
    b_ptr,          # float32[C_out] or dummy if HAS_BIAS=False
    y_ptr,          # float32[N, C_out, D_out, H_out, W_out]
    N,              # batch size
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    M,              # total output positions = N * D_out * H_out * W_out
    C_out,          # number of output channels
    C_IN: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    K_TOTAL: tl.constexpr,  # C_IN * K_D * K_H * K_W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Program IDs for tiling over M (output positions) and N (output channels)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile offsets
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M          # [BLOCK_M]
    mask_n = n_offsets < C_out      # [BLOCK_N]

    # Decode flattened M index -> (n_img, od, oh, ow)
    tmp = m_offsets
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n_img = tmp // D_out

    # Accumulator: [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute constants for K-dimension decoding
    R_spatial = K_D * K_H * K_W       # spatial kernel size
    R_hw = K_H * K_W

    # Loop over K dimension in BLOCK_K chunks
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]
        k_mask = k_offsets < K_TOTAL

        # Decode k_offsets -> (ci, kd, kh, kw)
        ci = k_offsets // R_spatial
        rem = k_offsets - ci * R_spatial
        kd = rem // R_hw
        rem = rem - kd * R_hw
        kh = rem // K_W
        kw = rem - kh * K_W

        # Broadcast to get input coordinates for each (m, k)
        # Shapes: od,oh,ow: [BLOCK_M]; kd,kh,kw: [BLOCK_K]
        id_ = od[:, None] * stride_d - pad_d + kd[None, :] * dil_d
        ih = oh[:, None] * stride_h - pad_h + kh[None, :] * dil_h
        iw = ow[:, None] * stride_w - pad_w + kw[None, :] * dil_w

        # Bounds check for input coordinates
        in_bounds = (
            (id_ >= 0) & (id_ < D_in) &
            (ih >= 0) & (ih < H_in) &
            (iw >= 0) & (iw < W_in)
        )

        # Full mask for input load
        mask_x = mask_m[:, None] & k_mask[None, :] & in_bounds

        # Flatten indices into x: [N, C_in, D_in, H_in, W_in] (contiguous)
        n_b = n_img[:, None]
        ci_b = ci[None, :]
        x_idx = (((n_b * C_IN + ci_b) * D_in + id_) * H_in + ih) * W_in + iw

        x_vals = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)
        x_vals_f32 = tl.cast(x_vals, tl.float32)  # [BLOCK_M, BLOCK_K]

        # Load weight tile as matrix [BLOCK_K, BLOCK_N]
        # w is treated as [C_out, K_TOTAL] row-major
        w_idx = k_offsets[:, None] + n_offsets[None, :] * K_TOTAL
        mask_w = k_mask[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        w_vals_f32 = tl.cast(w_vals, tl.float32)  # [BLOCK_K, BLOCK_N]

        # Block GEMM: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_vals_f32, w_vals_f32, allow_tf32=True)

    # Add bias if present: broadcast over BLOCK_M
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + n_offsets, mask=mask_n, other=0.0)  # [BLOCK_N]
        bias_vals_f32 = tl.cast(bias_vals, tl.float32)
        acc += bias_vals_f32[None, :]

    # Store results back to y: [N, C_out, D_out, H_out, W_out]
    out_DHW = D_out * H_out * W_out
    out_HW = H_out * W_out

    base_out_m = (
        n_img * (C_out * out_DHW) +
        od * out_HW +
        oh * W_out +
        ow
    )  # [BLOCK_M]

    y_idx = base_out_m[:, None] + n_offsets[None, :] * out_DHW
    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_idx, acc, mask=mask_y)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    # Fallback for grouped convs or non-fp32 dtypes
    if groups != 1 or x.dtype != torch.float32 or weight.dtype != torch.float32:
        return torch.nn.functional.conv3d(x, weight, bias, stride, padding, dilation, groups)

    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

    # Ensure contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_d, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels mismatch."

    # Normalize stride / padding / dilation to 3-tuples
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

    # Output dimensions
    D_out = (D_in + 2 * pad_d - dil_d * (K_d - 1) - 1) // stride_d + 1
    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    M = N * D_out * H_out * W_out
    y = torch.empty((N, C_out, D_out, H_out, W_out),
                    device=x.device,
                    dtype=x.dtype)

    K_total = C_in * K_d * K_h * K_w

    # Tile sizes (powers of two)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    has_bias = bias is not None
    b_ptr = bias if bias is not None else weight  # dummy pointer if no bias

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv3d_fwd_gemm_kernel[grid](
        x, weight, b_ptr, y,
        N, D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        M, C_out,
        C_IN=C_in,
        K_D=K_d,
        K_H=K_h,
        K_W=K_w,
        K_TOTAL=K_total,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=2,
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
