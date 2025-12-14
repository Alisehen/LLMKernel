import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr,          # (N, C_in, H, W)
    w_ptr,          # (C_out, K_total) where K_total = C_in * KH * KW
    b_ptr,          # (C_out,) or dummy
    y_ptr,          # (N, C_out, OH, OW)
    N, C_in, H, W,
    C_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    OH, OW,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Flattened output positions: M = N * OH * OW
    P = N * OH * OW

    pid_m = tl.program_id(axis=0)  # over output positions
    pid_n = tl.program_id(axis=1)  # over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Map offs_m -> (n, oh, ow)
    OHOW = OH * OW
    n_idxs = offs_m // OHOW
    rem = offs_m % OHOW
    oh_idxs = rem // OW
    ow_idxs = rem % OW

    n_b = n_idxs[:, None]   # [BM, 1]
    oh_b = oh_idxs[:, None] # [BM, 1]
    ow_b = ow_idxs[:, None] # [BM, 1]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension: all (channel, kh, kw)
    K_total = C_in * KH * KW
    KK = KH * KW

    # Iterate over K in BLOCK_K chunks
    for k_base in range(0, K_total, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < K_total

        # Decompose k into (ic, kh, kw)
        ic = offs_k // KK
        rem_k = offs_k % KK
        kh = rem_k // KW
        kw = rem_k % KW

        ic_b = ic[None, :]   # [1, BK]
        kh_b = kh[None, :]   # [1, BK]
        kw_b = kw[None, :]   # [1, BK]

        # Compute input coordinates
        ih = oh_b * stride_h - pad_h + kh_b * dil_h  # [BM, BK]
        iw = ow_b * stride_w - pad_w + kw_b * dil_w  # [BM, BK]

        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        # Mask for valid A loads
        mask_a = mask_m[:, None] & mask_k[None, :] & in_bounds

        # Flattened input index: ((n * C_in + ic) * H + ih) * W + iw
        nc = n_b * C_in + ic_b        # [BM, BK]
        nch = nc * H + ih             # [BM, BK]
        nchw = nch * W + iw           # [BM, BK]

        a = tl.load(x_ptr + nchw, mask=mask_a, other=0.0)

        # Weights tile: w_ptr is (C_out, K_total) row-major
        # B has shape [BK, BN] with indices [k, oc]
        w_ptrs = w_ptr + offs_n[None, :] * K_total + offs_k[:, None]
        mask_w = mask_n[None, :] & mask_k[:, None]
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate partial matmul
        acc += tl.dot(a, b)

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
        acc += bias_vals[None, :]

    # Store result to y: (N, C_out, OH, OW)
    oc_b = offs_n[None, :]  # [1, BN]
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
    High-performance 2D convolution via implicit GEMM using Triton.
    Assumes NCHW layout and groups == 1.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype, "x and weight must have same dtype"
    assert groups == 1, "This Triton kernel currently supports groups == 1 only"

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
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w * groups, "Inconsistent in_channels and groups"

    # Output spatial dimensions (same as PyTorch)
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    # Allocate output
    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    # Contiguous inputs
    x_contig = x.contiguous()
    w_mat = weight.contiguous().view(C_out, -1)  # (C_out, K_total)
    if bias is not None:
        b_contig = bias.contiguous()
    else:
        # Dummy tensor; will not be used when HAS_BIAS=False
        b_contig = x_contig.view(-1)

    # Launch grid
    P = N * OH * OW
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_implicit_gemm_kernel[grid](
        x_contig,
        w_mat,
        b_contig,
        y,
        N,
        C_in,
        H,
        W,
        C_out,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        OH,
        OW,
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated Conv2d via implicit-GEMM.
    Parameters are stored in an internal nn.Conv2d module for easy initialization
    and state_dict compatibility, but the forward path uses the Triton kernel.
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
