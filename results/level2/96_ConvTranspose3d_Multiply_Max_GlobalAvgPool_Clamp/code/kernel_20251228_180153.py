# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helper implementations for missing Triton functions
# ---------------------------------------------------------------------------

@triton.jit
def tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def tl_tanh(x):
    e_pos = tl.exp(x)
    e_neg = tl.exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)


@triton.jit
def tl_silu(x):
    return x * tl_sigmoid(x)


@triton.jit
def tl_gelu(x):
    # Approximate GELU
    k = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    return 0.5 * x * (1.0 + tl_tanh(k * (x + c * x * x * x)))


@triton.jit
def tl_mish(x):
    # mish(x) = x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(x))
    return x * tl_tanh(sp)


# ---------------------------------------------------------------------------
# Autotuned Triton kernel (aggressively optimized for 4090)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Conservative baseline (required)
        triton.Config({'BLOCK_W': 64},  num_warps=4, num_stages=2),
        # Slightly larger tile, still conservative
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        # High-parallelism, latency-hiding config (favored on 4090 if regs allow)
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=3),
        # Large tile for wide W, high BW utilization
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
    ],
    key=['W_pool'],  # tune per effective pooled width
)
@triton.jit
def scale_maxpool3d_globalavg_clamp_kernel(
    x_ptr, o_ptr,
    N, C,
    D_pool: tl.constexpr, H_pool: tl.constexpr, W_pool: tl.constexpr,
    stride_p_d, stride_p_h, stride_p_w,
    stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
    stride_n_out, stride_c_out,
    scale, inv_S_pool,
    clamp_min, clamp_max,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused operation per (n, c) slice:

      y[n, c] = clamp(
                    mean_{d2,h2,w2} max_{kd,kh,kw} ( x_scaled[...] ),
                    clamp_min, clamp_max
                )

    - No intermediate global stores: only a single final store to o_ptr.
    - All intermediate values (max, partial sums) stay in registers.
    """

    # Program id -> (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    # Base pointer for this (n, c) slice
    base_nc = x_ptr + n * stride_n_in + c * stride_c_in

    # Precompute pooled -> input strides (fewer integer ops in inner loops)
    step_pd = stride_p_d * stride_d_in  # advance d2 by one pooled step
    step_ph = stride_p_h * stride_h_in  # advance h2 by one pooled step
    step_pw = stride_p_w * stride_w_in  # advance w2 by one pooled step

    # Offsets along W for this program's vector of BLOCK_W lanes
    offs_w = tl.arange(0, BLOCK_W)

    # Scalar accumulator over entire pooled volume
    total = 0.0

    # Iterate over pooled spatial positions
    # (kept as runtime loops; work is dominated by K_D*K_H*K_W inner loops)
    for d2 in range(0, D_pool):
        d_base = base_nc + d2 * step_pd

        for h2 in range(0, H_pool):
            dh_base = d_base + h2 * step_ph

            # Vectorized traversal along pooled W
            for w_start in range(0, W_pool, BLOCK_W):
                pw = w_start + offs_w
                mask_w = pw < W_pool  # shared mask for all fused ops in this block

                # Pointer to the first element of each pooled output position
                base = dh_base + pw * step_pw

                # Max accumulator over pooling window for each lane
                max_val = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

                # Pooling window loops: fully unrolled via tl.static_range
                for kd in tl.static_range(0, K_D):
                    dd = kd * stride_d_in
                    for kh in tl.static_range(0, K_H):
                        hh = kh * stride_h_in
                        base_kdh = base + dd + hh
                        for kw in tl.static_range(0, K_W):
                            ptr = base_kdh + kw * stride_w_in
                            v = tl.load(ptr, mask=mask_w, other=-float("inf"))
                            v = v * scale
                            max_val = tl.maximum(max_val, v)

                # Zero out masked lanes (tail handling) before reduction
                max_val = tl.where(mask_w, max_val, 0.0)

                # Accumulate block sum into scalar total
                block_sum = tl.sum(max_val, axis=0)
                total += block_sum

    # Global average pooling over all pooled positions
    mean = total * inv_S_pool

    # Clamp to [clamp_min, clamp_max]
    mean = tl.minimum(mean, clamp_max)
    mean = tl.maximum(mean, clamp_min)

    # Output layout: (N, C, 1, 1, 1); single global store per output element
    out_ptr = o_ptr + n * stride_n_out + c * stride_c_out
    tl.store(out_ptr, mean)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

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

    # Precompute reciprocal of S_pool to avoid division in kernel
    inv_S_pool = 1.0 / float(S_pool)

    # 1D grid over (N * C), covering output tensor dims
    grid = lambda META: (max(1, N * C),)

    scale_maxpool3d_globalavg_clamp_kernel[grid](
        x, out,
        N, C,
        D_pool, H_pool, W_pool,
        stride_p_d, stride_p_h, stride_p_w,
        stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out,
        float(scale), float(inv_S_pool),
        float(clamp_min), float(clamp_max),
        K_D=k_d,
        K_H=k_h,
        K_W=k_w,
    )
    return out


# ---------------------------------------------------------------------------
# Model using the fused Triton kernel
# ---------------------------------------------------------------------------

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
