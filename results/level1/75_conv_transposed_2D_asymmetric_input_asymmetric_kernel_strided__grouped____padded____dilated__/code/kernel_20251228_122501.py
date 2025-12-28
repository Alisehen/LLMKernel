import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_implicit_gemm_kernel(
    x_ptr,           # float32[N, C_in, H_in, W_in]
    w_gemm_ptr,      # float32[groups, K_total, C_out_per_group]
    b_ptr,           # float32[C_out] or dummy
    out_ptr,         # float32[N, C_out, H_out, W_out]
    N,
    C_in,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    kH,
    kW,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    groups,
    C_in_per_group,
    C_out_per_group,
    M_total,   # N * H_out * W_out
    K_total,   # C_in_per_group * kH * kW
    M_tiles,   # ceil_div(M_total, BLOCK_M)
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implicit-GEMM implementation of ConvTranspose2d (grouped, strided, dilated, padded).

    Computes:
      out[n, oc, oy, ox] = sum_{ic, ky, kx} x[n, ic, iy, ix] * w[ic, oc_in_group, ky, kx]
    with the conv-transpose geometry.

    We treat:
      M dimension: m = n * (H_out * W_out) + oy * W_out + ox
      N dimension: oc within each group
      K dimension: (ky, kx, ic_in_group) flattened.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Decode group id and tile id in M
    gid = pid_m // M_tiles
    m_tile_id = pid_m - gid * M_tiles

    # Tile of output channels within the group
    oc_in_group_start = pid_n * BLOCK_N
    oc_in_group = oc_in_group_start + tl.arange(0, BLOCK_N)
    mask_n = oc_in_group < C_out_per_group

    # Global output channel indices
    oc_global = gid * C_out_per_group + oc_in_group

    # Tile of M = N * H_out * W_out
    m_start = m_tile_id * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M_total

    # Decode m_offsets -> (n_idx, oy, ox)
    HW_out = H_out * W_out
    n_idx = m_offsets // HW_out
    rem = m_offsets - n_idx * HW_out
    oy = rem // W_out
    ox = rem - oy * W_out

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Optional bias addition
    if has_bias:
        # Bias is [C_out]
        bias_ptrs = b_ptr + oc_global
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
        acc += bias[None, :]

    # Base pointer for this group's weights in GEMM layout
    # w_gemm_ptr layout: [groups, K_total, C_out_per_group]
    # Flatten group dimension: [groups * K_total * C_out_per_group]
    w_group_ptr = w_gemm_ptr + gid * (K_total * C_out_per_group)

    # Broadcasted per-row values
    n_bc = n_idx[:, None]
    oy_bc = oy[:, None]
    ox_bc = ox[:, None]

    # Loop over K dimension in BLOCK_K chunks
    for k0 in range(0, K_total, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_ids < K_total

        # ----- Load weight tile B: [BLOCK_K, BLOCK_N] -----
        # w_gemm[g, k, oc_in_group] with layout [K_total, C_out_per_group] per group
        b_ptrs = w_group_ptr + k_ids[:, None] * C_out_per_group + oc_in_group[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]
        B = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # ----- Compute A tile indices and load A: [BLOCK_M, BLOCK_K] -----
        # Decode K index -> (ky, kx, ic_in_group) consistent with weight packing:
        # k = ((ky * kW + kx) * C_in_per_group) + ic_in_group
        ic_in_group = k_ids % C_in_per_group
        tmp = k_ids // C_in_per_group
        ky = tmp // kW
        kx = tmp - ky * kW

        ky_bc = ky[None, :]
        kx_bc = kx[None, :]
        ic_global = gid * C_in_per_group + ic_in_group
        ic_global_bc = ic_global[None, :]

        # Geometric mapping for conv-transpose:
        # iy_unscaled = oy + pad_h - ky * dil_h
        # ix_unscaled = ox + pad_w - kx * dil_w
        iy_unscaled = oy_bc + pad_h - ky_bc * dil_h
        ix_unscaled = ox_bc + pad_w - kx_bc * dil_w

        iy = iy_unscaled // stride_h
        ix = ix_unscaled // stride_w

        rem_y = iy_unscaled - iy * stride_h
        rem_x = ix_unscaled - ix * stride_w

        valid = (
            (rem_y == 0)
            & (rem_x == 0)
            & (iy >= 0)
            & (iy < H_in)
            & (ix >= 0)
            & (ix < W_in)
        )

        row_mask = mask_m[:, None]
        col_mask = k_mask[None, :]
        mask_a = valid & row_mask & col_mask

        # x index: ((n * C_in + ic_global) * H_in + iy) * W_in + ix
        x_ptrs = x_ptr + (((n_bc * C_in + ic_global_bc) * H_in + iy) * W_in + ix)
        A = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # ----- FMA via matmul -----
        acc += tl.dot(A, B, allow_tf32=True)

    # ----- Store output tile -----
    oc_global_bc = oc_global[None, :]
    out_ptrs = out_ptr + (
        ((n_bc * C_out + oc_global_bc) * H_out + oy_bc) * W_out + ox_bc
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    """
    x:      [N, C_in, H_in, W_in]
    weight: [C_in, C_out_per_group, kH, kW]  (nn.ConvTranspose2d layout)
    bias:   [C_out] or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, kH, kW = weight.shape
    assert C_in_w == C_in
    assert C_in % groups == 0, "C_in must be divisible by groups"
    C_in_per_group = C_in // groups
    C_out = C_out_per_group * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # PyTorch conv_transpose2d output size formula (no output_padding)
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + 1

    out = torch.empty(
        (N, C_out, H_out, W_out), device=x.device, dtype=x.dtype
    )

    # Pre-pack weights into GEMM-friendly layout:
    #   [groups, C_in_per_group, C_out_per_group, kH, kW]
    # -> [groups, kH, kW, C_in_per_group, C_out_per_group]
    # -> [groups, K_total, C_out_per_group], K_total = kH * kW * C_in_per_group
    weight_view = weight.view(groups, C_in_per_group, C_out_per_group, kH, kW)
    weight_gemm = (
        weight_view.permute(0, 3, 4, 1, 2)
        .contiguous()
        .view(groups, kH * kW * C_in_per_group, C_out_per_group)
    )

    M_total = N * H_out * W_out
    K_total = C_in_per_group * kH * kW

    # Tiling parameters
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    M_tiles = triton.cdiv(M_total, BLOCK_M)
    N_tiles_per_group = triton.cdiv(C_out_per_group, BLOCK_N)

    grid = (
        groups * M_tiles,
        N_tiles_per_group,
    )

    has_bias = 1 if bias is not None else 0
    b_ptr = bias if bias is not None else out  # dummy pointer when no bias

    conv_transpose2d_implicit_gemm_kernel[grid](
        x,
        weight_gemm,
        b_ptr,
        out,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        C_in_per_group,
        C_out_per_group,
        M_total,
        K_total,
        M_tiles,
        has_bias=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated replacement for nn.ConvTranspose2d with the same constructor
    and behavior (no output_padding handled here).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        return triton_conv_transpose2d(
            x,
            w,
            b,
            stride=self.conv_transpose2d.stride,
            padding=self.conv_transpose2d.padding,
            dilation=self.conv_transpose2d.dilation,
            groups=self.conv_transpose2d.groups,
        )
