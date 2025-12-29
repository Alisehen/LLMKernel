import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_maxpool_gemm_kernel(
    x_ptr,               # float* [N, C_in, H, W]
    w_ptr,               # float* [C_out, C_in, K, K]
    conv_bias_ptr,       # float* [C_out]
    extra_bias_ptr,      # float* [C_out, 1, 1] (treated as [C_out])
    y_ptr,               # float* [N, C_out, H_pool, W_pool]
    N, C_in, H, W,
    C_out,
    H_out, W_out,
    H_pool, W_pool,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    scaling_factor,
    KERNEL_SIZE: tl.constexpr,
    POOL_KERNEL: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_p = tl.program_id(0)   # tiles over pooled spatial positions
    pid_oc = tl.program_id(1)  # tiles over output channels
    pid_n = tl.program_id(2)   # batch index

    # Output-channel tile
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < C_out

    # Pooled spatial tiles
    P_total = H_pool * W_pool
    offs_p_local = tl.arange(0, BLOCK_P)               # [0..BLOCK_P)
    p_global_block = pid_p * BLOCK_P + offs_p_local    # [BLOCK_P]
    mask_p = p_global_block < P_total                  # [BLOCK_P]

    # Flattened conv positions inside this pooled-tile:
    # Each pooled output has P_inner = POOL_KERNEL*POOL_KERNEL conv positions
    P_inner = POOL_KERNEL * POOL_KERNEL
    S_len = BLOCK_P * P_inner

    # S index over [0 .. BLOCK_P * P_inner)
    offs_s = tl.arange(0, BLOCK_P * P_inner)           # [S_len)
    p_idx = offs_s // P_inner                          # [S_len], local pooled index in this block
    inner_idx = offs_s % P_inner                       # [S_len], index inside pooling window

    # Global pooled index for each S element
    p_global_for_s = pid_p * BLOCK_P + p_idx           # [S_len]
    mask_s = p_global_for_s < P_total                  # [S_len]

    # Convert pooled index -> pooled (h_pool, w_pool)
    hp = p_global_for_s // W_pool                      # [S_len]
    wp = p_global_for_s % W_pool                       # [S_len]

    # Offset inside pooling window
    ih = inner_idx // POOL_KERNEL                      # [S_len]
    iw = inner_idx % POOL_KERNEL                       # [S_len]

    # Corresponding conv output coordinates
    conv_h = hp * POOL_STRIDE + ih                     # [S_len]
    conv_w = wp * POOL_STRIDE + iw                     # [S_len]

    # Total reduction dimension of conv
    K_total = C_in * KERNEL_SIZE * KERNEL_SIZE

    # Accumulator for convolution result over [OC_tile, S_len]
    acc = tl.zeros((BLOCK_OC, BLOCK_P * P_inner), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)          # [BLOCK_K]
        mask_k = offs_k < K_total

        # Decode flattened K index -> (ci, kh, kw)
        k_hw = KERNEL_SIZE * KERNEL_SIZE
        ci = offs_k // k_hw                               # [BLOCK_K]
        kk = offs_k % k_hw
        kh = kk // KERNEL_SIZE
        kw = kk % KERNEL_SIZE

        # Load weight block: [BLOCK_OC, BLOCK_K]
        w_ptrs = (
            w_ptr
            + offs_oc[:, None] * stride_wn
            + ci[None, :] * stride_wc
            + kh[None, :] * stride_wkh
            + kw[None, :] * stride_wkw
        )
        w_block = tl.load(
            w_ptrs,
            mask=mask_oc[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Compute input coordinates for this K-block and all S positions
        ci_b = ci[:, None]                      # [BLOCK_K, 1]
        kh_b = kh[:, None]                      # [BLOCK_K, 1]
        kw_b = kw[:, None]                      # [BLOCK_K, 1]

        h_in = conv_h[None, :] + kh_b           # [BLOCK_K, S_len]
        w_in = conv_w[None, :] + kw_b           # [BLOCK_K, S_len]

        # Input pointers: [BLOCK_K, S_len]
        x_ptrs = (
            x_ptr
            + pid_n * stride_xn
            + ci_b * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )
        x_block = tl.load(
            x_ptrs,
            mask=mask_k[:, None] & mask_s[None, :],
            other=0.0,
        )

        # GEMM-style update: [OC, BLOCK_K] @ [BLOCK_K, S_len] -> [OC, S_len]
        acc += tl.dot(w_block, x_block, allow_tf32=True)

    # Add convolution bias per output channel
    conv_bias = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + conv_bias[:, None]

    # Tanh activation: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    tmp = 2.0 * acc
    e = tl.exp(tmp)
    tanh_out = (e - 1.0) / (e + 1.0)

    # Scaling
    acc = tanh_out * scaling_factor

    # Reshape to [BLOCK_OC, BLOCK_P, P_inner] to perform pooling
    acc_3d = tl.reshape(acc, (BLOCK_OC, BLOCK_P, P_inner))

    # Max-pooling over P_inner for each pooled position
    acc_max = tl.full((BLOCK_OC, BLOCK_P), -1e30, dtype=tl.float32)
    for inner in range(0, P_inner):
        vals = acc_3d[:, :, inner]
        acc_max = tl.maximum(acc_max, vals)

    # Add extra bias [C_out, 1, 1] -> [C_out]
    extra_bias = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc_max = acc_max + extra_bias[:, None]

    # Store final output: [N, C_out, H_pool, W_pool]
    hp_block = p_global_block // W_pool   # [BLOCK_P]
    wp_block = p_global_block % W_pool    # [BLOCK_P]

    y_ptrs = (
        y_ptr
        + pid_n * stride_yn
        + offs_oc[:, None] * stride_yc
        + hp_block[None, :] * stride_yh
        + wp_block[None, :] * stride_yw
    )
    out_mask = mask_oc[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc_max, mask=out_mask)


def fused_conv_tanh_scale_bias_maxpool(x, weight, conv_bias, extra_bias, scaling_factor, pool_kernel_size):
    # Ensure contiguity for simple, fast indexing
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    extra_bias = extra_bias.contiguous()

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    KERNEL_SIZE = weight.shape[2]

    # Conv output spatial size (no padding, stride=1, dilation=1)
    H_out = H - KERNEL_SIZE + 1
    W_out = W - KERNEL_SIZE + 1

    POOL_KERNEL = pool_kernel_size
    POOL_STRIDE = pool_kernel_size

    # MaxPool2d output size: floor((H_out - Kp)/Sp + 1)
    H_pool = (H_out - POOL_KERNEL) // POOL_STRIDE + 1
    W_pool = (W_out - POOL_KERNEL) // POOL_STRIDE + 1

    y = torch.empty((N, C_out, H_pool, W_pool), device=x.device, dtype=x.dtype)

    P_total = H_pool * W_pool

    def grid(meta):
        return (
            triton.cdiv(P_total, meta['BLOCK_P']),
            triton.cdiv(C_out, meta['BLOCK_OC']),
            N,
        )

    fused_conv_tanh_scale_bias_maxpool_gemm_kernel[grid](
        x, weight, conv_bias, extra_bias, y,
        N, C_in, H, W,
        C_out,
        H_out, W_out,
        H_pool, W_pool,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        float(scaling_factor),
        KERNEL_SIZE=KERNEL_SIZE,
        POOL_KERNEL=POOL_KERNEL,
        POOL_STRIDE=POOL_STRIDE,
        BLOCK_P=16,      # power-of-2 tile over pooled spatial positions
        BLOCK_OC=64,     # power-of-2 tile over output channels
        BLOCK_K=32,      # power-of-2 tile over K dimension (C_in * KERNEL_SIZE^2)
    )
    return y


class ModelNew(nn.Module):
    """
    Fused Triton implementation of:
      Conv2d (no padding, stride=1) -> tanh -> scaling -> bias add -> MaxPool2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Conv parameters (match nn.Conv2d layout: [out_channels, in_channels, kH, kW])
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        # Extra bias with shape (out_channels, 1, 1)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        return fused_conv_tanh_scale_bias_maxpool(
            x, self.weight, self.conv_bias, self.bias, self.scaling_factor, self.pool_kernel_size
        )
