# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Final micro-tuning for Ada (4090, CC 8.9).
# Kernel is memory-bound (dram ~84% peak, SM ~37%, active warps ~57%):
#  - Keep tiling/grid exactly as-is (great L2 behavior).
#  - Slightly increase occupancy / latency hiding via num_warps.
#  - Keep num_stages low/moderate to avoid excess registers.
#  - Use 3–6 nearby configs; always include original baseline config.
conv3d_configs = [
    # --- Original baseline config (must keep) ---
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=2,
    ),

    # Same tile, more warps to increase occupancy and hide DRAM latency
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=8,
        num_stages=2,
    ),

    # Same tile, more warps + one extra pipeline stage (within ±1)
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
        num_warps=8,
        num_stages=3,
    ),

    # More N, less M – good when OC_per_group large, M small
    triton.Config(
        {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=2,
    ),

    # More M, less N – good when M large, OC_per_group small
    triton.Config(
        {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
        num_warps=4,
        num_stages=2,
    ),

    # Larger K tile to reduce loop iterations; keep stages moderate
    triton.Config(
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
        num_warps=4,
        num_stages=3,
    ),
]


@triton.autotune(
    configs=conv3d_configs,
    key=['M_TOTAL', 'OC_per_group'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr,        # *x_dtype  [N, C_in, D_in, H_in, W_in]
    w_ptr,        # *w_dtype  [C_out, C_in/groups, KD, KH, KW]
    b_ptr,        # *bias_dtype  [C_out] (may be unused when HAS_BIAS=False)
    y_ptr,        # *y_dtype  [N, C_out, D_out, H_out, W_out]
    N, C_in,
    D_in, H_in, W_in,
    C_out,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    IC_per_group,
    OC_per_group,
    KD, KH, KW,
    groups,
    M_TOTAL,                       # = N * D_out * H_out * W_out
    K_TOTAL: tl.constexpr,         # = IC_per_group * KD * KH * KW
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,         # tile over M = N*D_out*H_out*W_out
    BLOCK_N: tl.constexpr,         # tile over OC_per_group
    BLOCK_K: tl.constexpr,         # tile over K_TOTAL
):
    # -----------------------------
    # Program IDs / tiling
    # -----------------------------
    pid_m = tl.program_id(axis=0)  # over M dimension (N * D_out * H_out * W_out)
    pid_n = tl.program_id(axis=1)  # over output channels within a group
    pid_g = tl.program_id(axis=2)  # over groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    m_mask = offs_m < M_TOTAL
    n_mask = offs_n < OC_per_group

    # -----------------------------
    # Decode linear M -> (n, d_out, h_out, w_out)
    # -----------------------------
    w_out_idx = offs_m % W_out
    tmp = offs_m // W_out
    h_out_idx = tmp % H_out
    tmp = tmp // H_out
    d_out_idx = tmp % D_out
    n_idx = tmp // D_out

    # Input base coords for the receptive field
    d_in_base = d_out_idx * stride_d - pad_d
    h_in_base = h_out_idx * stride_h - pad_h
    w_in_base = w_out_idx * stride_w - pad_w

    # Group offsets
    c_in_group_offset = pid_g * IC_per_group
    oc_group_offset = pid_g * OC_per_group

    # -----------------------------
    # Strides assuming NCDHW contiguous
    # -----------------------------
    x_strideN = C_in * D_in * H_in * W_in
    x_strideC = D_in * H_in * W_in
    x_strideD = H_in * W_in
    x_strideH = W_in
    x_strideW = 1

    y_strideN = C_out * D_out * H_out * W_out
    y_strideC = D_out * H_out * W_out
    y_strideD = H_out * W_out
    y_strideH = W_out
    y_strideW = 1

    # Global OC indices for this tile (within full C_out)
    oc_inner = offs_n
    oc = oc_group_offset + oc_inner
    oc_mask = n_mask

    # -----------------------------
    # Accumulator
    # -----------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Broadcasted output indices for pointer arithmetic
    n_b = n_idx[:, None]
    d_out_b = d_out_idx[:, None]
    h_out_b = h_out_idx[:, None]
    w_out_b = w_out_idx[:, None]

    # Precompute constant stride in weight tensor: [C_out, K_TOTAL]
    w_strideOC = K_TOTAL

    # -----------------------------
    # Main K loop: implicit GEMM over (ci, kd, kh, kw)
    # -----------------------------
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        tl.multiple_of(offs_k, BLOCK_K)

        # Decode linear K -> (ci, kd, kh, kw)
        kw = offs_k % KW
        tmpk = offs_k // KW
        kh = tmpk % KH
        tmpk = tmpk // KH
        kd = tmpk % KD
        ci = tmpk // KD

        # Broadcast to [BLOCK_M, BLOCK_K]
        kd_b = kd[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]
        ci_b = ci[None, :]

        # Compute input coordinates for this (M,K) tile
        d_in = d_in_base[:, None] + kd_b * dil_d
        h_in = h_in_base[:, None] + kh_b * dil_h
        w_in = w_in_base[:, None] + kw_b * dil_w
        c_in = c_in_group_offset + ci_b

        # Bounds check for input
        in_bounds = (
            m_mask[:, None]
            & k_mask[None, :]
            & (d_in >= 0)
            & (d_in < D_in)
            & (h_in >= 0)
            & (h_in < H_in)
            & (w_in >= 0)
            & (w_in < W_in)
        )

        # Input pointers [BLOCK_M, BLOCK_K]
        in_ptrs = (
            x_ptr
            + n_b * x_strideN
            + c_in * x_strideC
            + d_in * x_strideD
            + h_in * x_strideH
            + w_in * x_strideW
        )

        # Load A tile; source dtype -> fp32 accumulate
        a = tl.load(in_ptrs, mask=in_bounds, other=0.0)
        a = a.to(tl.float32)

        # Weight pointers [BLOCK_K, BLOCK_N]
        k_b = offs_k[:, None]
        oc_b = oc[None, :]

        w_ptrs = w_ptr + oc_b * w_strideOC + k_b
        w_bounds = k_mask[:, None] & oc_mask[None, :]

        b = tl.load(w_ptrs, mask=w_bounds, other=0.0)
        b = b.to(tl.float32)

        # GEMM accumulate
        acc += tl.dot(a, b)

    # -----------------------------
    # Epilogue: bias + store
    # -----------------------------
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc, mask=oc_mask, other=0.0).to(tl.float32)
        acc += bias_vals[None, :]

    out_ptrs = (
        y_ptr
        + n_b * y_strideN
        + oc[None, :] * y_strideC
        + d_out_b * y_strideD
        + h_out_b * y_strideH
        + w_out_b * y_strideW
    )
    out_mask = m_mask[:, None] & oc_mask[None, :]

    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# -----------------------------
# Wrapper: launch configuration
# -----------------------------
def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """
    Triton implementation of Conv3d using implicit GEMM.
    Expects x in shape [N, C_in, D_in, H_in, W_in] and contiguous tensors.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_grp, KD, KH, KW = weight.shape

    # Handle stride / padding / dilation (int or 3-tuple)
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation

    IC_per_group = C_in_grp
    assert IC_per_group * groups == C_in, "Inconsistent in_channels/groups"
    assert C_out % groups == 0, "out_channels must be divisible by groups"
    OC_per_group = C_out // groups

    # Output shape (same formula as PyTorch Conv3d)
    D_out = (D_in + 2 * pad_d - dil_d * (KD - 1) - 1) // stride_d + 1
    H_out = (H_in + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    y = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=torch.float32,  # kernel accumulates & outputs in fp32
    )

    M_total = N * D_out * H_out * W_out
    K_TOTAL = IC_per_group * KD * KH * KW

    # Grid: [M tiles, OC/group tiles, groups]
    def grid(meta):
        return (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(OC_per_group, meta['BLOCK_N']),
            groups,
        )

    conv3d_implicit_gemm_kernel[grid](
        x, weight, bias if bias is not None else weight,
        y,
        N, C_in,
        D_in, H_in, W_in,
        C_out,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        IC_per_group,
        OC_per_group,
        KD, KH, KW,
        groups,
        M_total,
        K_TOTAL=K_TOTAL,
        HAS_BIAS=(bias is not None),
    )

    return y


# -----------------------------
# Module wrapper
# -----------------------------
class ModelNew(nn.Module):
    """
    Triton-accelerated Conv3d module using an implicit-GEMM Triton kernel.
    Tuned for Ada (RTX 4090) with aggressive latency hiding and memory
    efficiency via autotuned blocking / warps / stages.
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
    ) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, 1),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            self.conv3d.stride,
            self.conv3d.padding,
            self.conv3d.dilation,
            self.conv3d.groups,
        )
