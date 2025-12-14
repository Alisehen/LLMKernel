import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 16},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["N", "C_out", "H", "W", "KH", "KW"],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr,          # *f32/f16, (N, C_in, H, W)
    w_ptr,          # *f32/f16, (C_out, K_total) where K_total = (C_in/groups)*KH*KW
    b_ptr,          # *f32/f16, (C_out,) or dummy
    y_ptr,          # *f32/f16, (N, C_out, OH, OW)
    N, C_in, H, W,
    C_out, K_total,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    OH, OW,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs:
    #   axis 0 -> tiles over output positions P = N * OH * OW
    #   axis 1 -> tiles over output channels within a group
    #   axis 2 -> groups
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_g = tl.program_id(axis=2)

    OC_per_group = C_out // groups
    IC_per_group = C_in // groups
    P = N * OH * OW

    # Offsets along output-position (M) and output-channel (N) dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    mask_m = offs_m < P
    mask_n = offs_n < OC_per_group

    # For masked-out positions, use a safe index 0 to keep pointers in-bounds
    offs_m_safe = tl.where(mask_m, offs_m, 0)

    # Map linear output index -> (n, oh, ow)
    OHOW = OH * OW
    n_idxs = offs_m_safe // OHOW
    rem = offs_m_safe % OHOW
    oh_idxs = rem // OW
    ow_idxs = rem % OW

    # Group channel offsets
    oc_group_offset = pid_g * OC_per_group
    ic_group_offset = pid_g * IC_per_group

    # Broadcasted versions of (n, oh, ow) for use in the K-loop
    n_b = n_idxs[:, None]   # [BM, 1]
    oh_b = oh_idxs[:, None] # [BM, 1]
    ow_b = ow_idxs[:, None] # [BM, 1]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KK = KH * KW

    # Iterate over K dimension (IC_per_group * KH * KW)
    for k_base in range(0, K_total, BLOCK_K):
        k_idxs = k_base + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = k_idxs < K_total
        # Clamp K indices to last valid position for out-of-range lanes
        k_safe = tl.where(mask_k, k_idxs, K_total - 1)

        # Decompose k index into (ic_rel, kh, kw)
        ic_rel = k_safe // KK
        rem_k = k_safe % KK
        kh = rem_k // KW
        kw = rem_k % KW

        ic = ic_group_offset + ic_rel  # [BK]

        # Broadcast for A (im2col) loads
        ic_b = ic[None, :]   # [1, BK]
        kh_b = kh[None, :]   # [1, BK]
        kw_b = kw[None, :]   # [1, BK]

        # Compute input spatial indices
        ih = oh_b * stride_h - pad_h + kh_b * dil_h  # [BM, BK]
        iw = ow_b * stride_w - pad_w + kw_b * dil_w  # [BM, BK]

        # Check in-bounds for input
        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        mask_a = mask_m[:, None] & mask_k[None, :] & in_bounds

        # Clamp spatial indices to keep pointers in-bounds even when masked-out
        ih_clamped = tl.where(ih < 0, 0, tl.where(ih >= H, H - 1, ih))
        iw_clamped = tl.where(iw < 0, 0, tl.where(iw >= W, W - 1, iw))

        # Compute flat input offsets: ((n * C_in + ic) * H + ih) * W + iw
        nc = n_b * C_in + ic_b           # [BM, BK]
        nch = nc * H + ih_clamped        # [BM, BK]
        nchw = nch * W + iw_clamped      # [BM, BK]

        a_ptrs = x_ptr + nchw
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Weights slice: w_ptr is (C_out, K_total)
        oc = oc_group_offset + offs_n    # [BN]
        oc_safe = tl.where(mask_n, oc, 0)
        oc_b = oc_safe[None, :]          # [1, BN]

        w_ptrs = w_ptr + (oc_b * K_total + k_safe[:, None])  # [BK, BN]
        mask_w = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate GEMM: (BM, BK) x (BK, BN) -> (BM, BN)
        acc += tl.dot(a, b)

    # Add bias if present
    oc = oc_group_offset + offs_n  # [BN]
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store result to y: (N, C_out, OH, OW)
    oc_safe = tl.where(mask_n, oc, 0)
    oc_b = oc_safe[None, :]             # [1, BN]

    out_offsets = ((n_b * C_out + oc_b) * OH + oh_b) * OW + ow_b  # [BM, BN]
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + out_offsets, acc, mask=mask_out)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """
    Conv2d via implicit GEMM using Triton.
    Semantics match torch.nn.functional.conv2d for NCHW layout.
    """

    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype, "x and weight must have same dtype"

    # Normalize hyperparameters to pairs
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    assert C_in == C_in_per_group * groups, "Inconsistent in_channels and groups"

    # Output spatial dimensions
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    # Allocate output
    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    # Prepare contiguous tensors
    x_contig = x.contiguous()
    # Flatten weight to (C_out, K_total)
    w_mat = weight.contiguous().view(C_out, -1)
    if bias is not None:
        b_contig = bias.contiguous()
    else:
        # Dummy tensor, won't be used when HAS_BIAS=False
        b_contig = x_contig.view(-1)

    P = N * OH * OW
    OC_per_group = C_out // groups
    K_total = w_mat.shape[1]

    # Grid function uses autotuned BLOCK_M/BLOCK_N from meta
    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(OC_per_group, meta["BLOCK_N"]),
            groups,
        )

    conv2d_implicit_gemm_kernel[grid](
        x_contig, w_mat, b_contig, y,
        N, C_in, H, W,
        C_out, K_total,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        OH, OW,
        HAS_BIAS=(bias is not None),
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated Conv2d using implicit-GEMM with autotuned tiling.
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
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv2d.weight
        b = self.conv2d.bias
        return triton_conv2d(
            x,
            w,
            b,
            stride=self.conv2d.stride,
            padding=self.conv2d.padding,
            dilation=self.conv2d.dilation,
            groups=self.conv2d.groups,
        )
