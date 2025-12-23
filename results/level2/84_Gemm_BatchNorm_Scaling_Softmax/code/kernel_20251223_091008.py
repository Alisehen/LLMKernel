# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------
# Optimized GEMM (Linear) kernel: y = x @ W^T + b
# Tuned for RTX 4090 (Ada) with aggressive tiling and
# register-pressure-aware num_warps / num_stages.
#
# Memory-fusion rule:
#   - Only ONE tl.store() for the output tile
#   - All intermediates (matmul + bias) stay in registers
# ---------------------------------------------
@triton.autotune(
    configs=[
        # High throughput for large M,N (compute-friendly, more warps)
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        # Conservative baseline (good occupancy, lower registers)
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_forward_kernel(
    a_ptr,  # [M, K], row-major
    b_ptr,  # [K, N] logical (transposed weight)
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id: tile in output space [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Hints for better codegen / vectorization
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Masks for M and N dimensions (broadcasted later with K)
    mask_m = offs_m < M  # (BLOCK_M,)
    mask_n = offs_n < N  # (BLOCK_N,)

    # Base pointers for this tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 for better precision and tensor-core friendly matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_indices = k + offs_k  # (BLOCK_K,)
        k_mask = k_indices < K  # (BLOCK_K,)

        # Full 2D masks for loads
        a_mask = mask_m[:, None] & k_mask[None, :]    # (BLOCK_M, BLOCK_K)
        b_mask = k_mask[:, None] & mask_n[None, :]    # (BLOCK_K, BLOCK_N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias add: broadcast over rows, using offs_n / mask_n for indexing
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Write back final result (single store site: no intermediate stores)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance replacement for:
        x = F.linear(x, weight, bias)  # (x @ weight.T + bias)

    Args:
        x:      [M, K]
        weight: [N, K]
        bias:   [N]
    Returns:
        y:      [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda

    # Ensure contiguous for predictable strides and aligned memory
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # x is [M, K] row-major
    stride_am, stride_ak = x.stride()

    # Interpret weight as logical [K, N] without material transpose:
    #   physical: weight [N, K], strides (K, 1)
    #   logical: B[k, n] = weight[n, k]
    stride_w0, stride_w1 = weight.stride()  # (N, K)
    stride_bk = stride_w1  # stride along K dim
    stride_bn = stride_w0  # stride along N dim

    stride_cm, stride_cn = y.stride()

    def grid(meta):
        grid_m = triton.cdiv(M, meta['BLOCK_M'])
        grid_n = triton.cdiv(N, meta['BLOCK_N'])
        return (max(1, grid_m), max(1, grid_n))

    linear_forward_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return y


# ---------------------------------------------
# Optimized fused scale + softmax over dim=1
# y[i, j] = softmax_j(scale * x[i, j])
#
# Memory-fusion constraints:
#   - Multiple loads from x are allowed
#   - NO intermediate stores: exactly one tl.store() site for y
#   - All temporaries (max, sum) stay in registers
#
# 3-pass algorithm, with 2 exp() per element:
#   1) Find row-wise max of scaled values
#   2) Compute row-wise sum of exp(x - max)
#   3) Compute final softmax values and store once
#
# This removes the store-then-load pattern and cuts global memory traffic
# at the cost of extra compute, which is favorable on a 4090.
# ---------------------------------------------
@triton.autotune(
    configs=[
        # Conservative baseline (good for wide range of N)
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        # Higher per-row parallelism
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        # Large tile for long rows, more warps to saturate memory
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def fused_scale_softmax_kernel(
    x_ptr, y_ptr,
    scale,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid: each program handles one row of the output
    row = tl.program_id(0)
    row_in_bounds = row < M

    offs = tl.arange(0, BLOCK_SIZE)

    # Row start pointers
    x_row_ptr = x_ptr + row * stride_xm
    y_row_ptr = y_ptr + row * stride_ym

    # -------------------------
    # Pass 1: row-wise max of scaled values
    # -------------------------
    row_max = -float('inf')
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs                      # (BLOCK_SIZE,)
        col_mask = cols < N                        # (BLOCK_SIZE,)
        mask = row_in_bounds & col_mask            # (BLOCK_SIZE,)

        x_block_ptrs = x_row_ptr + cols * stride_xn
        # Load with -inf for OOB so they don't affect max
        x_vals = tl.load(x_block_ptrs, mask=mask, other=-float('inf'))
        x_scaled = x_vals * scale
        # Explicitly ensure OOB lanes are -inf
        x_scaled = tl.where(mask, x_scaled, -float('inf'))
        block_max = tl.max(x_scaled, axis=0)       # scalar
        row_max = tl.maximum(row_max, block_max)

    # -------------------------
    # Pass 2: compute sum of exp(x - row_max)
    # -------------------------
    row_sum = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        col_mask = cols < N
        mask = row_in_bounds & col_mask

        x_block_ptrs = x_row_ptr + cols * stride_xn
        x_vals = tl.load(x_block_ptrs, mask=mask, other=-float('inf'))
        x_scaled = x_vals * scale
        x_scaled = tl.where(mask, x_scaled, -float('inf'))

        exp_x = tl.exp(x_scaled - row_max)
        # Only in-bounds lanes contribute to the sum
        exp_x = tl.where(mask, exp_x, 0.0)

        block_sum = tl.sum(exp_x, axis=0)
        row_sum += block_sum

    inv_row_sum = 1.0 / row_sum

    # -------------------------
    # Pass 3: compute final softmax and store ONCE
    # -------------------------
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        col_mask = cols < N
        mask = row_in_bounds & col_mask

        x_block_ptrs = x_row_ptr + cols * stride_xn
        y_block_ptrs = y_row_ptr + cols * stride_yn

        x_vals = tl.load(x_block_ptrs, mask=mask, other=0.0)
        x_scaled = x_vals * scale
        exp_x = tl.exp(x_scaled - row_max)
        y_vals = exp_x * inv_row_sum

        # Single store site for output tensor y
        tl.store(y_block_ptrs, y_vals, mask=mask)


def fused_scale_softmax(x: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
    """
    High-performance replacement for:
        x = scale * x
        x = softmax(x, dim=1)

    Args:
        x:           [M, N]
        scale_param: scalar tensor (shape (1,) or broadcastable to x)
    Returns:
        y:           [M, N]
    """
    assert x.is_cuda and scale_param.is_cuda
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    # Use scalar scale for best performance
    scale = float(scale_param.item())

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    def grid(meta):
        # One program per row; ensure at least one program for degenerate M
        return (max(1, M),)

    fused_scale_softmax_kernel[grid](
        x, y,
        scale,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
    )
    return y


# ---------------------------------------------
# Model with Triton-accelerated Linear + scale*Softmax
# BatchNorm is kept as in PyTorch to preserve training semantics
# ---------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        # Keep module structure/parameter names for easy state_dict loading
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))

    def forward(self, x):
        # Expect x to be on CUDA for Triton kernels
        x = fused_linear(x, self.gemm.weight, self.gemm.bias)
        x = self.bn(x)
        x = fused_scale_softmax(x, self.scale)
        return x
