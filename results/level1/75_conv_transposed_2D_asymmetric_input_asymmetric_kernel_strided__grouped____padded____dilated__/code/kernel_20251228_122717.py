# <complete ModelNew code with optimized Triton kernels>

import torch, torch.nn as nn, triton, triton.language as tl


@triton.autotune(
    configs=[
        # Higher-occupancy, lighter K tile: better latency hiding on 4090
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            num_warps=4,
            num_stages=2,
        ),
        # Heavier K tile for more math/byte when registers allow
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger M tile for tall M (N*H_out*W_out) shapes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "C_OUT", "GROUPS"],
)
@triton.jit
def conv_transpose2d_igemm_kernel(
    x_ptr,        # (N, C_IN, H_IN, W_IN)
    w_ptr,        # (C_IN, C_OUT_PER_GROUP, KH, KW)
    b_ptr,        # (C_OUT,) or dummy
    out_ptr,      # (N, C_OUT, H_OUT, W_OUT)
    N, C_IN, H_IN, W_IN,
    C_OUT, H_OUT, W_OUT,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    M,            # total number of output elements along (N*H_OUT*W_OUT)
    GROUPS,       # only used for autotune key; not needed inside kernel
    K_TOTAL: tl.constexpr,          # C_IN_PER_GROUP * KH * KW
    C_IN_PER_GROUP: tl.constexpr,
    C_OUT_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program ids:
    #   pid_m : tiles over M = N * H_OUT * W_OUT
    #   pid_n : tiles over output channels within each group (BLOCK_N)
    #   pid_g : group id
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_g = tl.program_id(axis=2)

    # Row indices (M dimension: N * H_OUT * W_OUT)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Decode (n, ho, wo) from linear M index
    HW_OUT = H_OUT * W_OUT
    n = offs_m // HW_OUT
    rem = offs_m % HW_OUT
    ho = rem // W_OUT
    wo = rem % W_OUT

    # Group id
    group_id = pid_g

    # Column indices (N dimension: C_OUT), organized as groups * blocks_per_group
    co_block_id = pid_n
    co_in_group_start = co_block_id * BLOCK_N
    offs_n_group = co_in_group_start + tl.arange(0, BLOCK_N)
    co_group_mask = offs_n_group < C_OUT_PER_GROUP

    # Global output channels
    co = group_id * C_OUT_PER_GROUP + offs_n_group
    co_mask = co_group_mask & (co < C_OUT)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # K loop: iterate over (ci_in_group, kh, kw) combinations
    # -------------------------------------------------------------------------
    for k0 in range(0, K_TOTAL, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K_TOTAL

        # Decode k_idx -> (ci_in_group, kh, kw)
        ci_g = k_idx // (KH * KW)
        remk = k_idx % (KH * KW)
        kh = remk // KW
        kw = remk % KW

        # Global input channel index for this group
        ci = group_id * C_IN_PER_GROUP + ci_g  # [BLOCK_K]

        # ---- Load weight tile (B matrix: [K, N]) ----
        ci_b = ci[:, None]              # [BLOCK_K, 1]
        co_b = offs_n_group[None, :]    # [1, BLOCK_N]
        kh_b = kh[:, None]              # [BLOCK_K, 1]
        kw_b = kw[:, None]              # [BLOCK_K, 1]

        w_ptrs = w_ptr + (((ci_b * C_OUT_PER_GROUP + co_b) * KH + kh_b) * KW + kw_b)
        w_mask = k_mask[:, None] & co_group_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        b_tile = w_vals.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # ---- Load input tile (A matrix: [M, K]) ----
        n_b = n[:, None]    # [BLOCK_M, 1]
        ho_b = ho[:, None]  # [BLOCK_M, 1]
        wo_b = wo[:, None]  # [BLOCK_M, 1]

        kh_b2 = kh[None, :]  # [1, BLOCK_K]
        kw_b2 = kw[None, :]  # [1, BLOCK_K]

        # Compute input coordinates (hi, wi) that map to (ho, wo) via deconv relation:
        # ho + PAD_H = hi * STRIDE_H + kh * DIL_H
        # wo + PAD_W = wi * STRIDE_W + kw * DIL_W
        num_h = ho_b + PAD_H - kh_b2 * DIL_H
        num_w = wo_b + PAD_W - kw_b2 * DIL_W

        hi = num_h // STRIDE_H
        wi = num_w // STRIDE_W

        # Validity masks for mapping
        mask_h_ge0 = num_h >= 0
        mask_w_ge0 = num_w >= 0
        mask_h_int = (num_h % STRIDE_H) == 0
        mask_w_int = (num_w % STRIDE_W) == 0
        mask_hi_range = (hi >= 0) & (hi < H_IN)
        mask_wi_range = (wi >= 0) & (wi < W_IN)

        pix_valid = (
            mask_h_ge0 & mask_h_int & mask_hi_range &
            mask_w_ge0 & mask_w_int & mask_wi_range
        )

        ci_b2 = ci[None, :]  # [1, BLOCK_K]

        x_ptrs = x_ptr + (((n_b * C_IN + ci_b2) * H_IN + hi) * W_IN + wi)
        mk_mask = m_mask[:, None] & k_mask[None, :]
        x_mask = mk_mask & pix_valid

        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        a_tile = x_vals.to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # ---- Matrix multiply accumulate ----
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    # -------------------------------------------------------------------------
    # Bias add
    # -------------------------------------------------------------------------
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + co, mask=co_mask, other=0.0)
        b_vals_f32 = b_vals.to(tl.float32)
        acc += b_vals_f32[None, :]

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------
    n_b = n[:, None]
    ho_b = ho[:, None]
    wo_b = wo[:, None]
    co_b_global = co[None, :]

    out_ptrs = out_ptr + (((n_b * C_OUT + co_b_global) * H_OUT + ho_b) * W_OUT + wo_b)
    out_mask = m_mask[:, None] & co_mask[None, :]

    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: tuple,
                            padding: tuple,
                            dilation: tuple,
                            groups: int) -> torch.Tensor:
    # Ensure CUDA + contiguous
    assert x.is_cuda and weight.is_cuda, "Input and weights must be CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Restrict to float32 for now (matches provided reference usage)
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, \
        "This Triton implementation currently supports only float32 tensors."

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, KH, KW = weight.shape
    assert C_in_w == C_in, "Weight C_in mismatch"

    C_out = C_out_per_group * groups

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Output spatial dimensions (PyTorch ConvTranspose2d formula)
    H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # GEMM-style dimensions
    M_total = N * H_out * W_out
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    K_total = C_in_per_group * KH * KW

    # Grid function depends on autotuned BLOCK sizes
    def grid(META):
        # 3D grid:
        #  - axis 0: tiles over M (N*H_out*W_out)
        #  - axis 1: tiles over output channels within each group
        #  - axis 2: groups
        return (
            triton.cdiv(M_total, META["BLOCK_M"]),
            triton.cdiv(C_out_per_group, META["BLOCK_N"]),
            groups,
        )

    conv_transpose2d_igemm_kernel[grid](
        x, weight, bias if bias is not None else out, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        M_total,
        groups,
        K_TOTAL=K_total,
        C_IN_PER_GROUP=C_in_per_group,
        C_OUT_PER_GROUP=C_out_per_group,
        KH=KH,
        KW=KW,
        HAS_BIAS=(bias is not None),
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for nn.ConvTranspose2d with arbitrary stride,
    padding, dilation and groups, using an implicit-GEMM formulation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple,
                 stride: tuple = (1, 1), padding: tuple = (0, 0),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        # Use PyTorch module only to hold parameters (weight/bias, groups, etc.)
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
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding
        dilation = self.conv_transpose2d.dilation
        groups = self.conv_transpose2d.groups
        return triton_conv_transpose2d(x, w, b, stride, padding, dilation, groups)
