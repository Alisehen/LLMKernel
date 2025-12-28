import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_scale_min_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_ob, stride_oc, stride_oh, stride_ow,
    scale,
    BLOCK_M: tl.constexpr,  # rows in im2col (output points)
    BLOCK_N: tl.constexpr,  # output channels per tile
    BLOCK_K: tl.constexpr,  # reduction dim (Ci*Kh*Kw) per tile
):
    pid_m = tl.program_id(0)

    # Total number of output points (B * Ho * Wo)
    N_points = B * Ho * Wo

    # Rows of the implicit im2col matrix handled by this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_points

    # Decode linear index into (b, y, x)
    hw = Ho * Wo
    b = offs_m // hw
    rem = offs_m % hw
    y = rem // Wo
    x = rem % Wo

    K_total = Ci * Kh * Kw
    KhKw = Kh * Kw

    # Initialize running minimum per output point
    inf = 3.4e38
    min_val = tl.full((BLOCK_M,), inf, tl.float32)

    # Loop over output channels in tiles of size BLOCK_N
    for oc_start in range(0, Co, BLOCK_N):
        oc_offsets = oc_start + tl.arange(0, BLOCK_N)
        mask_n = oc_offsets < Co

        # GEMM accumulator for this (M, N) tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Reduction over K = Ci * Kh * Kw using tiled matmul
        for k_start in range(0, K_total, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < K_total

            # Decode k_offsets -> (ci, ky, kx)
            ci = k_offsets // KhKw
            remk = k_offsets % KhKw
            ky = remk // Kw
            kx = remk % Kw

            # Compute input coordinates (iy, ix) for each (m, k)
            iy = y[:, None] + ky[None, :]
            ix = x[:, None] + kx[None, :]

            # Pointers into input tensor x: [B, Ci, Hi, Wi]
            x_ptrs = (
                x_ptr
                + b[:, None] * stride_xb
                + ci[None, :] * stride_xc
                + iy * stride_xh
                + ix * stride_xw
            )
            mask_x = mask_m[:, None] & mask_k[None, :]
            x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Pointers into weight tensor w: [Co, Ci, Kh, Kw]
            w_ptrs = (
                w_ptr
                + oc_offsets[None, :] * stride_wco
                + ci[:, None] * stride_wci
                + ky[:, None] * stride_wkh
                + kx[:, None] * stride_wkw
            )
            mask_w = mask_k[:, None] & mask_n[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Matmul accumulate: (M, K) @ (K, N) -> (M, N)
            acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        # Add bias and scale
        bias_vals = tl.load(bias_ptr + oc_offsets, mask=mask_n, other=0.0)
        acc = (acc + bias_vals[None, :]) * scale

        # Exclude invalid output-channel columns from min reduction
        acc = tl.where(mask_n[None, :], acc, inf)

        # Tile-wise min over channels and update running minimum
        tile_min = tl.min(acc, 1)
        min_val = tl.minimum(min_val, tile_min)

    # Store result: output shape [B, 1, Ho, Wo]
    out_ptrs = (
        out_ptr
        + b * stride_ob
        + 0 * stride_oc
        + y * stride_oh
        + x * stride_ow
    )
    tl.store(out_ptrs, min_val, mask=mask_m)


def conv_scale_min_triton(x, weight, bias, scale_factor):
    """
    x:       [B, Ci, Hi, Wi]
    weight:  [Co, Ci, Kh, Kw]
    bias:    [Co]
    Returns: [B, 1, Ho, Wo] with Ho/Wo as in PyTorch conv2d (stride=1, padding=0, dilation=1).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    B, Ci, Hi, Wi = x.shape
    Co, Ci_w, Kh, Kw = weight.shape
    assert Ci == Ci_w, "In-channel mismatch between input and weight"
    assert bias.shape[0] == Co, "Bias must match out_channels"

    # Conv2d output spatial size for stride=1, padding=0, dilation=1
    Ho = Hi - Kh + 1
    Wo = Wi - Kw + 1
    assert Ho > 0 and Wo > 0, "Invalid kernel size for given input"

    out = torch.empty((B, 1, Ho, Wo), device=x.device, dtype=x.dtype)

    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wco, stride_wci, stride_wkh, stride_wkw = weight.stride()
    stride_ob, stride_oc, stride_oh, stride_ow = out.stride()

    N_points = B * Ho * Wo

    # Tuned block sizes for the given problem shape
    BLOCK_M = 64   # output points
    BLOCK_N = 64   # output channels per tile
    BLOCK_K = 32   # reduction dimension tile

    def grid(meta):
        return (triton.cdiv(N_points, meta["BLOCK_M"]),)

    conv_scale_min_gemm_kernel[grid](
        x, weight, bias, out,
        B, Ci, Hi, Wi, Co, Kh, Kw, Ho, Wo,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_wco, stride_wci, stride_wkh, stride_wkw,
        stride_ob, stride_oc, stride_oh, stride_ow,
        float(scale_factor),
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
        Conv2d -> scale -> min over channel dim (keepdim=True)
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        return conv_scale_min_triton(x, self.weight, self.bias, self.scale_factor)
