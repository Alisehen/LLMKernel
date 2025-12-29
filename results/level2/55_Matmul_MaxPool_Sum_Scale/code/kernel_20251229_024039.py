# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ======================
# Linear kernel: C = A @ B + bias
# ======================

@triton.autotune(
    configs=[
        # Main configs (balanced register usage vs throughput)
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Conservative, very low register pressure fallback
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: C = A @ B + bias, where
      A: (M, K)
      B: (K, N)
      bias: (N,)
      C: (M, N)
    """

    # 2D grid for output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in output space
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for output region
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_out = mask_m[:, None] & mask_n[None, :]

    # K offsets (within tile)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32, even if inputs are FP16/BF16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Matrix multiply-accumulate
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias add (load once per tile)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store result (downcast handled by Triton if c_ptr dtype < fp32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_out)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)  (nn.Linear.weight layout)
    bias: (N,)
    returns: (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Triton kernel expects B as (K, N)
    w_t = weight.contiguous().T  # (K, N)

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        # Ensure grid dims are at least 1 to satisfy launch constraints
        grid_m = max(1, triton.cdiv(M, meta['BLOCK_M']))
        grid_n = max(1, triton.cdiv(N, meta['BLOCK_N']))
        return (grid_m, grid_n)

    linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ======================
# MaxPool + Sum + Scale kernel
# ======================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_P': 512}, num_warps=8, num_stages=1),
        # Conservative fallback for high register pressure cases
        triton.Config({'BLOCK_P': 128}, num_warps=4, num_stages=1),
    ],
    key=['P'],
)
@triton.jit
def maxpool_sum_scale_kernel(
    y_ptr, out_ptr,
    M, P, N,
    stride_ym, stride_yn,
    scale,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """
    y: (M, N)
    P = N // KERNEL_SIZE  (number of pooling windows)

    For each row m:
        out[m] = scale * sum_{p=0..P-1} max_{i=0..KERNEL_SIZE-1} y[m, p*KERNEL_SIZE + i]

    Each program handles a single row and processes windows in tiles of BLOCK_P.
    This keeps register usage low: only BLOCK_P temporaries per tile.
    """

    pid_m = tl.program_id(0)
    mask_m = pid_m < M

    # Per-row accumulator (scalar in FP32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # Fixed window indices within a tile
    offs_p_base = tl.arange(0, BLOCK_P)

    # Iterate over all pooling windows in tiles of BLOCK_P
    for p in range(0, P, BLOCK_P):
        offs_p = p + offs_p_base  # [BLOCK_P]
        mask_p = offs_p < P

        # Max values for current tile of windows (for this single row)
        max_vals = tl.full((BLOCK_P,), -1.0e30, dtype=tl.float32)

        # Compute max over each pooling window
        for i in range(KERNEL_SIZE):
            offs_n = offs_p * KERNEL_SIZE + i  # [BLOCK_P]

            ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
            vals = tl.load(
                ptrs,
                mask=mask_m & mask_p,
                other=-1.0e30,
            )
            max_vals = tl.maximum(max_vals, vals)

        # Zero-out invalid windows (tail in P)
        max_vals = tl.where(mask_p, max_vals, 0.0)

        # Accumulate sum over the pooled dimension for this tile
        row_sum += tl.sum(max_vals, axis=0)

    # Scaling + store
    out_val = row_sum * scale
    tl.store(out_ptr + pid_m, out_val, mask=mask_m)


def fused_maxpool_sum_scale(y: torch.Tensor, kernel_size: int, scale_factor: float) -> torch.Tensor:
    """
    y: (M, N)
    returns: (M,)
    """
    assert y.is_cuda
    M, N = y.shape
    kernel_size = int(kernel_size)
    P = N // kernel_size  # number of pooling windows

    out = torch.empty((M,), device=y.device, dtype=y.dtype)

    def grid(meta):
        # One program per row; ensure at least 1 program for launch correctness
        return (max(1, M),)

    maxpool_sum_scale_kernel[grid](
        y, out,
        M, P, N,
        y.stride(0), y.stride(1),
        float(scale_factor),
        KERNEL_SIZE=kernel_size,
    )
    return out


# ======================
# Model
# ======================

class ModelNew(nn.Module):
    """
    Triton-optimized version:
      x -> Linear -> MaxPool1d(kernel_size, stride=kernel_size) over feature dim
        -> sum over feature dim -> scale_factor
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        y = fused_linear(x, self.weight, self.bias)                             # (B, out_features)
        out = fused_maxpool_sum_scale(y, self.kernel_size, self.scale_factor)   # (B,)
        return out
