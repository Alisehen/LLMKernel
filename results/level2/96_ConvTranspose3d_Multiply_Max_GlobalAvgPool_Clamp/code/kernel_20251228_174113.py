import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_maxpool3d_globalavg_clamp_kernel(
    x_ptr, o_ptr,
    N, C,
    D_in, H_in, W_in,
    D_pool, H_pool, W_pool,
    S_pool,                      # S_pool = D_pool * H_pool * W_pool
    stride_p_d, stride_p_h, stride_p_w,
    k_d, k_h, k_w,
    stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
    stride_n_out, stride_c_out,
    scale,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
):
    # One program instance per (n, c) pair
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C

    # Base pointer for this (n, c) slice in the input
    base_nc = x_ptr + n * stride_n_in + c * stride_c_in

    # Vector accumulator over spatial positions (S_pool)
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Iterate over pooled output positions in chunks of BLOCK_S
    for offset in range(0, S_pool, BLOCK_S):
        offs = offset + tl.arange(0, BLOCK_S)
        mask_o = offs < S_pool

        # Compute pooled output indices (d2, h2, w2) from flattened index
        w2 = offs % W_pool
        tmp = offs // W_pool
        h2 = tmp % H_pool
        d2 = tmp // H_pool

        # Corresponding input window start indices
        d0 = d2 * stride_p_d
        h0 = h2 * stride_p_h
        w0 = w2 * stride_p_w

        # Base pointer for each pooled window start (vectorized over BLOCK_S)
        base_in = (
            base_nc
            + d0 * stride_d_in
            + h0 * stride_h_in
            + w0 * stride_w_in
        )

        # Max accumulator for this block's pooled outputs
        max_val = tl.full((BLOCK_S,), -float("inf"), dtype=tl.float32)

        # Loop over pooling window
        for kd in range(0, k_d):
            for kh in range(0, k_h):
                for kw in range(0, k_w):
                    ptr = base_in + kd * stride_d_in + kh * stride_h_in + kw * stride_w_in
                    v = tl.load(ptr, mask=mask_o, other=0.0)
                    v = v.to(tl.float32)
                    v = v * scale
                    max_val = tl.maximum(max_val, v)

        # Zero-out invalid positions before accumulation
        max_val = tl.where(mask_o, max_val, 0.0)
        acc += max_val

    # Reduce accumulator to a single scalar sum over all pooled positions
    total = tl.sum(acc, axis=0)
    mean = total / S_pool

    # Clamp to [clamp_min, clamp_max]
    mean = tl.minimum(mean, clamp_max)
    mean = tl.maximum(mean, clamp_min)

    # Output layout: (N, C, 1, 1, 1), contiguous
    out_ptr = o_ptr + n * stride_n_out + c * stride_c_out
    tl.store(out_ptr, mean)


def fused_scale_maxpool3d_globalavg_clamp(
    x: torch.Tensor,
    scale: float,
    kernel_size,
    clamp_min: float,
    clamp_max: float,
):
    """
    Fused operation:
      y = clamp( GlobalAvgPool3d( MaxPool3d( x * scale ) ), clamp_min, clamp_max )

    Args:
        x: (N, C, D, H, W), float32, CUDA, contiguous
        scale: scalar multiplier
        kernel_size: int or (k_d, k_h, k_w) for MaxPool3d
        clamp_min, clamp_max: scalar clamp bounds

    Returns:
        out: (N, C, 1, 1, 1), float32, CUDA
    """
    x = x.contiguous()
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "Kernel expects float32 tensor"

    if isinstance(kernel_size, int):
        k_d = k_h = k_w = kernel_size
    else:
        k_d, k_h, k_w = kernel_size

    # MaxPool3d defaults: stride = kernel_size, padding = 0, dilation = 1, ceil_mode = False
    stride_p_d = k_d
    stride_p_h = k_h
    stride_p_w = k_w
    pad_d = pad_h = pad_w = 0
    dilation_d = dilation_h = dilation_w = 1

    N, C, D_in, H_in, W_in = x.shape

    def _out_dim(L_in, k, s, p, d):
        # PyTorch pooling formula (ceil_mode=False)
        return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

    D_pool = _out_dim(D_in, k_d, stride_p_d, pad_d, dilation_d)
    H_pool = _out_dim(H_in, k_h, stride_p_h, pad_h, dilation_h)
    W_pool = _out_dim(W_in, k_w, stride_p_w, pad_w, dilation_w)
    S_pool = D_pool * H_pool * W_pool

    # Output of global average pooling: (N, C, 1, 1, 1)
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in = x.stride()
    stride_n_out, stride_c_out, _, _, _ = out.stride()

    grid = lambda META: (N * C,)

    scale_maxpool3d_globalavg_clamp_kernel[grid](
        x, out,
        N, C,
        D_in, H_in, W_in,
        D_pool, H_pool, W_pool,
        S_pool,
        stride_p_d, stride_p_h, stride_p_w,
        k_d, k_h, k_w,
        stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out,
        float(scale),
        float(clamp_min), float(clamp_max),
        BLOCK_S=128,
    )
    return out


class ModelNew(nn.Module):
    """
    Replacement model using Triton kernel for:
      - scalar multiply + MaxPool3d + global average pooling + clamp
    ConvTranspose3d is kept from PyTorch for correctness and simplicity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.scale = float(scale)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        # x: (N, C_in, D, H, W)
        x = self.conv_transpose(x)  # (N, C_out, D1, H1, W1)
        # Fused: scale -> MaxPool3d -> global average -> clamp
        x = fused_scale_maxpool3d_globalavg_clamp(
            x,
            self.scale,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
        )
        return x
