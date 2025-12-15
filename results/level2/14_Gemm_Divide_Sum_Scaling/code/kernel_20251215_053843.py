import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced default
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        # Higher parallelism along M when batch is large
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        # Conservative fallback to reduce register pressure
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def gemv_scaled_kernel(
    x_ptr,          # *dtype, shape [M, K]
    wsum_ptr,       # *dtype, shape [K]
    y_ptr,          # *dtype, shape [M]
    M, K,           # int32
    stride_xm, stride_xk,
    stride_ym,
    alpha,          # float32 scalar (scaling_factor / 2)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulate in fp32 for stability/perf, even if inputs are fp16/bf16
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        mask_k = k_offsets < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        wsum_ptrs = wsum_ptr + k_offsets

        # Load tile of x and corresponding slice of wsum
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wsum = tl.load(wsum_ptrs, mask=mask_k, other=0.0)

        # Upcast to fp32 for accumulation
        x = tl.cast(x, tl.float32)
        wsum = tl.cast(wsum, tl.float32)

        # Fused multiply-accumulate over K tile
        acc += tl.sum(x * wsum[None, :], axis=1)

    # Final scaling
    acc *= alpha

    y_ptrs = y_ptr + offs_m * stride_ym
    tl.store(y_ptrs, acc, mask=mask_m)


def fused_matmul_div_sum_scale(x: torch.Tensor, weight: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Computes:
        y = scaling_factor * sum_h ( (x @ weight.T)[..., h] / 2 )
      = (scaling_factor / 2) * x @ weight.sum(dim=0)
    Returns y with shape (batch_size, 1).
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA device"
    assert x.dtype == weight.dtype, "x and weight must have the same dtype"

    M, K = x.shape
    H, K_w = weight.shape
    assert K_w == K, "Incompatible shapes between x and weight"

    # Precompute column-wise sum over hidden dimension on GPU-optimized torch op
    weight_sum = weight.sum(dim=0).contiguous()

    # Overall scaling: divide by 2, then multiply by scaling_factor
    alpha = float(scaling_factor) * 0.5

    y = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)

    gemv_scaled_kernel[grid](
        x,
        weight_sum,
        y.view(-1),  # [M]
        M,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        alpha,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated model using:
        sum_h ( (x @ W^T)[b, h] / 2 ) * s
      = (s / 2) * x[b] · (sum_h W[h])
    Reduced to a single GEMV with aggressive tiling/autotuning.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_matmul_div_sum_scale(x, self.weight, self.scaling_factor)
