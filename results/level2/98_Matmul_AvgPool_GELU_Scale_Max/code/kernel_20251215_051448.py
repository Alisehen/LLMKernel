import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# -----------------------------------------------------------------------------
# Kernel 1: Matmul (A @ W^T) + bias + GELU + scale, then per-tile row-wise max
# -----------------------------------------------------------------------------
# Produces partial maxima over BLOCK_N feature tiles:
#   partial_max[pid_n, m] = max_{n in tile(pid_n)} GELU( (A @ W^T + b) * scale )[m, n]
# Final row-wise max is done in a separate reduction kernel.
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_stages=2,
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_gelu_scale_blockmax_kernel(
    a_ptr,          # [M, K], dtype=float16
    w_ptr,          # [N, K], dtype=float16  (pooled weights)
    b_ptr,          # [N],    dtype=float16  (pooled bias)
    partial_max_ptr,  # [num_n_tiles, M], dtype=float32
    M, N, K,
    stride_am, stride_ak,   # strides for A
    stride_wm, stride_wk,   # strides for W
    stride_pm_n, stride_pm_m,  # strides for partial_max [num_n_tiles, M]
    scale,                  # float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling over [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator for C[m, n] tile (float32 for high-precision accumulation)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    k = 0
    while k < K:
        k_offsets = k + offs_k
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (
            offs_m[:, None] * stride_am
            + k_offsets[None, :] * stride_ak
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Load W tile arranged as [BLOCK_K, BLOCK_N]
        # W is [N, K] row-major: w[n, k] => base + n*stride_wm + k*stride_wk
        w_ptrs = w_ptr + (
            offs_n[None, :] * stride_wm
            + k_offsets[:, None] * stride_wk
        )
        w = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # GEMM partial on Tensor Cores (fp16 inputs, fp32 accumulation)
        acc += tl.dot(a, w)

        k += BLOCK_K

    # Fused epilogue on the same [M, N] tile coordinates: bias + GELU + scale
    # ----------------------------------------------------------------------
    # Bias add (broadcast along M)
    bias = tl.load(
        b_ptr + offs_n,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)  # compute in fp32
    acc = acc + bias[None, :]

    # GELU approximation in fp32 (tanh-based, implemented via exp)
    x = acc
    x3 = x * x * x
    t = 0.7978845608028654 * (x + 0.044715 * x3)  # √(2/π) ≈ 0.79788456
    u = tl.exp(2.0 * t)
    tanh_t = (u - 1.0) / (u + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_t)

    # Scale
    gelu_scaled = gelu * scale

    # Row-wise max over this N-tile, with N masking
    neg_inf = -float("inf")
    masked_vals = tl.where(mask_n[None, :], gelu_scaled, neg_inf)
    block_max = tl.max(masked_vals, axis=1)  # [BLOCK_M]

    # Store partial maxima: shape [num_n_tiles, M]
    pm_ptrs = partial_max_ptr + pid_n * stride_pm_n + offs_m * stride_pm_m
    tl.store(pm_ptrs, block_max, mask=mask_m)


# -----------------------------------------------------------------------------
# Kernel 2: Final row-wise max over N tiles
# -----------------------------------------------------------------------------

@triton.jit
def row_max_kernel(
    partial_max_ptr,  # [num_n_tiles, M], float32
    out_ptr,          # [M], float32
    M,
    num_n_tiles,
    stride_pm_n, stride_pm_m,
    stride_outm,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    neg_inf = -float("inf")
    max_vals = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)

    # Reduce over N tiles
    tile_id = 0
    while tile_id < num_n_tiles:
        pm_ptrs = partial_max_ptr + tile_id * stride_pm_n + offs_m * stride_pm_m
        vals = tl.load(pm_ptrs, mask=mask_m, other=neg_inf)
        max_vals = tl.maximum(max_vals, vals)
        tile_id += 1

    out_ptrs = out_ptr + offs_m * stride_outm
    tl.store(out_ptrs, max_vals, mask=mask_m)


# -----------------------------------------------------------------------------
# Python wrapper: pooling weights, launching Triton kernels
# -----------------------------------------------------------------------------

def fused_matmul_avgpool_gelu_scale_max(x, weight, bias, pool_kernel_size, scale_factor):
    """
    Fused implementation of:
        y = x @ W^T + b
        y = AvgPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size) along features
        y = GELU(y)
        y = y * scale_factor
        out = max(y, dim=1)

    Using the identity:
        avg_pool(x @ W^T + b)  ==  x @ W_pooled^T + b_pooled

    where W_pooled and b_pooled are averages of weight/bias over pooling groups.

    This implementation:
      * Pools weights/bias on the host.
      * Uses a high-performance Triton GEMM kernel (fp16 Tensor Cores, fp32 accum).
      * Fuses bias + GELU + scale into the GEMM epilogue.
      * Computes row-wise max in two stages:
          - per-N-tile partial max in the GEMM kernel
          - final reduction over tiles in a second Triton kernel
    """
    assert x.is_cuda, "Input must be on CUDA device"
    device = x.device

    # Use float16 for matmul to exploit Tensor Cores; keep reduction/output in float32
    compute_dtype = torch.float16

    # Pool weights and biases in float32 for better numerical stability
    w_f32 = weight.float()
    b_f32 = bias.float()
    x_orig_dtype = x.dtype

    M, K = x.shape
    N = w_f32.shape[0]
    k = pool_kernel_size

    # Output length after AvgPool1d(kernel=k, stride=k, padding=0)
    # L_out = floor((N - k) / k) + 1
    N_pool = (N - k) // k + 1
    assert N_pool > 0, "Invalid configuration: out_features < pool_kernel_size"

    N_eff = N_pool * k  # Only the first N_eff features participate in pooling

    w_eff = w_f32[:N_eff]   # [N_eff, K]
    b_eff = b_f32[:N_eff]   # [N_eff]

    # Pool weights and biases along the output-feature dimension
    # w_eff: [N_pool, k, K] -> mean over k -> [N_pool, K]
    # b_eff: [N_pool, k]    -> mean over k -> [N_pool]
    w_pooled = w_eff.view(N_pool, k, K).mean(dim=1).contiguous()
    b_pooled = b_eff.view(N_pool, k).mean(dim=1).contiguous()

    # Cast to compute dtype (fp16) for GEMM
    x_mat = x.to(compute_dtype).contiguous()
    w_pooled_mat = w_pooled.to(compute_dtype).contiguous()
    b_pooled_mat = b_pooled.to(compute_dtype).contiguous()

    # Number of N tiles for BLOCK_N = 128 (fixed across autotune configs)
    BLOCK_N = 128
    num_n_tiles = triton.cdiv(N_pool, BLOCK_N)

    # Partial maxima buffer: [num_n_tiles, M], float32
    partial_max = torch.empty(
        (num_n_tiles, M),
        device=device,
        dtype=torch.float32,
    )

    # Launch matmul_gelu_scale_blockmax_kernel
    def grid_mm(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N_pool, meta['BLOCK_N']),
        )

    matmul_gelu_scale_blockmax_kernel[grid_mm](
        x_mat, w_pooled_mat, b_pooled_mat, partial_max,
        M, N_pool, K,
        x_mat.stride(0), x_mat.stride(1),
        w_pooled_mat.stride(0), w_pooled_mat.stride(1),
        partial_max.stride(0), partial_max.stride(1),
        float(scale_factor),
    )

    # Final row-wise max over N tiles
    out = torch.empty((M,), device=device, dtype=torch.float32)
    BLOCK_M = 128

    grid_max = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

    row_max_kernel[grid_max](
        partial_max, out,
        M,
        num_n_tiles,
        partial_max.stride(0), partial_max.stride(1),
        out.stride(0),
        BLOCK_M=BLOCK_M,
    )

    # Cast back to original input dtype if needed
    if x_orig_dtype != torch.float32:
        return out.to(x_orig_dtype)
    return out


# -----------------------------------------------------------------------------
# nn.Module wrapper
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized version of the reference model implementing:
        Matmul -> AvgPool1d -> GELU -> Scale -> Max

    Uses:
      * Weight/bias pooling on the host
      * fp16 Tensor Core GEMM with fused GELU+scale
      * Two-stage row-wise max reduction for high parallelism
    """

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        # Match nn.Linear(in_features, out_features) semantics
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # Initialization similar to nn.Linear
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_matmul_avgpool_gelu_scale_max(
            x, self.weight, self.bias, self.pool_kernel_size, self.scale_factor
        )
