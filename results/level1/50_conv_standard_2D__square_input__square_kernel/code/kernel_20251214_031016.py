# optimized Triton code

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Optimized Conv2D NCHW forward kernel (fp32) for Ada (RTX 4090)
# - Tiled over (output positions, output channels)
# - Tiled & streamed over K = C_in * KH * KW to reduce register pressure
# - Autotuned over BLOCK_M / BLOCK_N / BLOCK_K with grid depending on META
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['N', 'C_in', 'H', 'W', 'C_out', 'OH', 'OW'],
)
@triton.jit
def conv2d_fwd_kernel(
    x_ptr,         # float*  [N, C_in, H, W]   (contiguous, NCHW)
    w_ptr,         # float*  [C_out, C_in, KH, KW] (contiguous)
    b_ptr,         # float*  [C_out]
    out_ptr,       # float*  [N, C_out, OH, OW]

    N, C_in, H, W,
    C_out, KH, KW,
    OH, OW,
    stride_h, stride_w,
    pad_h, pad_w,

    K_TOTAL: tl.constexpr,   # = C_in * KH * KW (compile-time for given weight shape)
    BLOCK_M: tl.constexpr,   # tile over output positions (N * OH * OW)
    BLOCK_N: tl.constexpr,   # tile over output channels
    BLOCK_K: tl.constexpr,   # tile over K with static loop
):
    # -----------------------------------------------------------------------
    # Program ids
    # -----------------------------------------------------------------------
    pid_m = tl.program_id(0)  # tile over output positions
    pid_n = tl.program_id(1)  # tile over output channels

    # -----------------------------------------------------------------------
    # Offsets for M (output positions) and N (output channels)
    # -----------------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    M_total = N * OH * OW
    mask_m = offs_m < M_total          # [BM]
    mask_n = offs_n < C_out            # [BN]

    # -----------------------------------------------------------------------
    # Decode linear output index m -> (n, oh, ow)
    # m in [0, N*OH*OW)
    # -----------------------------------------------------------------------
    tmp = offs_m
    n_idx = tmp // (OH * OW)
    rem = tmp % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # -----------------------------------------------------------------------
    # Initialize accumulator for [BM, BN] tile
    # -----------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Helpful broadcasts
    n_b = n_idx[:, None]        # [BM,1]
    oh_b = oh[:, None]          # [BM,1]
    ow_b = ow[:, None]          # [BM,1]
    mask_m_b = mask_m[:, None]  # [BM,1]

    # -----------------------------------------------------------------------
    # Stream over K dimension in tiles of BLOCK_K
    # -----------------------------------------------------------------------
    KHW = KH * KW

    for k_base in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)        # [BK]
        k_mask = offs_k < K_TOTAL                      # [BK]

        # Map linear k -> (ic, kh, kw)
        ic = offs_k // KHW                             # [BK]
        rem_k = offs_k % KHW                           # [BK]
        kh = rem_k // KW                               # [BK]
        kw = rem_k % KW                                # [BK]

        ic_b = ic[None, :]                             # [1,BK]
        kh_b = kh[None, :]                             # [1,BK]
        kw_b = kw[None, :]                             # [1,BK]

        # Input coordinates for each (m, k)
        ih = oh_b * stride_h - pad_h + kh_b            # [BM,BK]
        iw = ow_b * stride_w - pad_w + kw_b            # [BM,BK]

        # Valid input mask for each (m, k)
        mask_in = (
            mask_m_b &
            (ih >= 0) & (ih < H) &
            (iw >= 0) & (iw < W) &
            k_mask[None, :]
        )

        # Linear offsets into input tensor: ((n * C_in + ic) * H + ih) * W + iw
        x_offsets = ((n_b * C_in + ic_b) * H + ih) * W + iw  # [BM,BK]
        a = tl.load(x_ptr + x_offsets, mask=mask_in, other=0.0)
        a = a.to(tl.float32)  # accumulate in fp32

        # -------------------------------------------------------------------
        # Load weights as [BK, BN] tile:
        # logical layout: weight[oc, ic, kh, kw]
        # flatten (ic, kh, kw) -> k in [0, K_TOTAL)
        # linear index = oc * K_TOTAL + k
        # -------------------------------------------------------------------
        offs_k_b = offs_k[:, None]                     # [BK,1]
        offs_n_b = offs_n[None, :]                     # [1,BN]

        mask_w = (offs_k_b < K_TOTAL) & (offs_n_b < C_out)

        w_offsets = offs_n_b * K_TOTAL + offs_k_b      # [BK,BN]
        b = tl.load(w_ptr + w_offsets, mask=mask_w, other=0.0)
        b = b.to(tl.float32)

        # -------------------------------------------------------------------
        # Matrix multiply accumulate: [BM,BK] x [BK,BN] -> [BM,BN]
        # -------------------------------------------------------------------
        acc += tl.dot(a, b)

    # -----------------------------------------------------------------------
    # Add bias: bias[oc]
    # -----------------------------------------------------------------------
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    acc += bias[None, :]  # broadcast over BM

    # -----------------------------------------------------------------------
    # Write back to output: out[n, oc, oh, ow]
    # index = ((n * C_out + oc) * OH + oh) * OW + ow
    # -----------------------------------------------------------------------
    n_out = n_idx[:, None]       # [BM,1]
    oh_out = oh[:, None]         # [BM,1]
    ow_out = ow[:, None]         # [BM,1]
    oc_out = offs_n[None, :]     # [1,BN]

    out_offsets = ((n_out * C_out + oc_out) * OH + oh_out) * OW + ow_out  # [BM,BN]
    mask_out = mask_m_b & (oc_out < C_out)

    tl.store(out_ptr + out_offsets, acc, mask=mask_out)


# ---------------------------------------------------------------------------
# Wrapper: grid calculation & kernel launch
# ---------------------------------------------------------------------------

def triton_conv2d_nchw(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    x:      [N, C_in, H, W],  contiguous, CUDA
    weight: [C_out, C_in, KH, KW], contiguous, CUDA
    bias:   [C_out], CUDA
    stride, padding: int (assumed symmetric for H/W)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported"
    assert bias is None or bias.dtype == torch.float32

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in_w == C_in, "Incompatible in_channels"

    stride_h = stride_w = stride
    pad_h = pad_w = padding

    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    # Precompute K_TOTAL as compile-time meta-parameter for the kernel.
    K_TOTAL = C_in * KH * KW

    # Grid: 2D over (output positions, output channels)
    M_total = N * OH * OW

    def grid(meta):
        return (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv2d_fwd_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out, KH, KW,
        OH, OW,
        stride_h, stride_w,
        pad_h, pad_w,
        K_TOTAL=K_TOTAL,
    )

    return out


# ---------------------------------------------------------------------------
# Model using the optimized Triton conv2d for the first layer
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super().__init__()
        # Same conv1 definition as the original model
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU fallback for safety
        if not x.is_cuda:
            return self.conv1(x)

        return triton_conv2d_nchw(
            x,
            self.conv1.weight,
            self.conv1.bias,
            stride=self.conv1.stride[0],
            padding=self.conv1.padding[0],
        )
