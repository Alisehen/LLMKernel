import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out,
    KH, KW,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    scale,
    BLOCK_M: tl.constexpr,  # tile over P = N * H_out * W_out
    BLOCK_N: tl.constexpr,  # tile over C_out (reduced)
    BLOCK_K: tl.constexpr,  # tile over K = C_in * KH * KW
):
    # Program id over flattened spatial positions P = N * H_out * W_out
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    P = N * H_out * W_out
    K = C_in * KH * KW
    HW_out = H_out * W_out

    # Mask for valid P
    mask_m = offs_m < P

    # Decode flattened M index -> (n, oh, ow)
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Initialize running minimum over C_out for each position in BLOCK_M
    INF = 1e30
    curr_min = tl.full((BLOCK_M,), INF, dtype=tl.float32)

    # Loop over output-channel tiles (reduction dimension)
    for oc0 in range(0, C_out, BLOCK_N):
        offs_n = oc0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < C_out

        # Accumulator for this [M, BLOCK_N] tile (FP32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Map flattened K index -> (ic, kh, kw)
            ic = offs_k // (KH * KW)
            rem_k = offs_k % (KH * KW)
            kh = rem_k // KW
            kw = rem_k % KW

            # -------- Load input tile A: [BLOCK_M, BLOCK_K] --------
            ih = oh_idx[:, None] + kh[None, :]
            iw = ow_idx[:, None] + kw[None, :]

            x_offsets = (
                n_idx[:, None] * stride_xn
                + ic[None, :] * stride_xc
                + ih * stride_xh
                + iw * stride_xw
            )
            x_ptrs = x_ptr + x_offsets
            mask_x = mask_m[:, None] & mask_k[None, :]
            x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # -------- Load weight tile B: [BLOCK_K, BLOCK_N] --------
            w_offsets = (
                offs_n[None, :] * stride_wn
                + ic[:, None] * stride_wc
                + kh[:, None] * stride_wkh
                + kw[:, None] * stride_wkw
            )
            w_ptrs = w_ptr + w_offsets
            mask_w = mask_k[:, None] & mask_n[None, :]
            w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Matrix multiply accumulate: [M,K] x [K,N] -> [M,N]
            acc += tl.dot(x_vals, w_vals)

        # -------- Fuse bias add & scale, then reduce over BLOCK_N --------
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        bias = bias.to(acc.dtype)
        acc = acc + bias[None, :]
        acc = acc * scale

        # Mask out invalid channels in this tile with +INF so they don't affect min
        acc = tl.where(mask_n[None, :], acc, INF)

        # Row-wise min over BLOCK_N -> shape [BLOCK_M]
        tile_min = tl.min(acc, axis=1)
        curr_min = tl.minimum(curr_min, tile_min)

    # -------- Store final per-position minimum (single output store) --------
    out_offsets = (
        n_idx * stride_on
        + oh_idx * stride_oh
        + ow_idx * stride_ow
    )
    out_ptrs = out_ptr + out_offsets
    tl.store(out_ptrs, curr_min, mask=mask_m)


def conv_scale_min_triton(x, weight, bias, scale_factor):
    """
    Fused Triton implementation of:
        y = Conv2d(x, weight, bias)
        y = y * scale_factor
        out = y.min(dim=1, keepdim=True)

    x:      [N, C_in, H, W]
    weight: [C_out, C_in, KH, KW]
    bias:   [C_out]
    out:    [N, 1, H_out, W_out]
    """
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight"

    # No padding, stride=1
    H_out = H - KH + 1
    W_out = W - KW + 1

    # Final output has single channel (min over C_out)
    out = torch.empty((N, 1, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wkh, stride_wkw = weight.stride()
    stride_on, stride_oc, stride_oh, stride_ow = out.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (triton.cdiv(P, meta["BLOCK_M"]),)

    # Tuned for Ada (4090); BLOCK_M large to amortize per-tile overhead
    conv2d_scale_min_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W, C_out,
        KH, KW,
        H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wkh, stride_wkw,
        stride_on, stride_oc, stride_oh, stride_ow,
        scale_factor,
        BLOCK_M=128,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Fused Triton model:
        y = Conv2d(x, weight, bias)
        y = y * scale_factor
        out = y.min(dim=1, keepdim=True)
    Implemented in a single kernel with only one global store of the final result.
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
        # Expect x: [N, C_in, H, W] on CUDA
        return conv_scale_min_triton(x, self.weight, self.bias, self.scale_factor)
