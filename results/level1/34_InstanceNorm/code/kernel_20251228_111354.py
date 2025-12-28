# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def instance_norm_single_pass_kernel(
    x_ptr,          # *f32 / *f16 / *bf16, [M, S] flattened
    y_ptr,          # same dtype as x, [M, S]
    M,              # int: number of (N, C) rows
    S,              # int: spatial size H*W
    eps,            # float32
    NUM_CHUNKS,     # int: number of chunks along S
    BLOCK_SIZE: tl.constexpr,  # power-of-2, e.g., 256
):
    pid = tl.program_id(axis=0)
    # grid is (M,), so 0 <= pid < M

    row = pid
    row_start = row * S

    offs = tl.arange(0, BLOCK_SIZE)

    # ------------------------------------------------------------------
    # Pass 1: compute per-row sum and sum of squares (no atomics)
    # ------------------------------------------------------------------
    sum_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sum_sq_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate over S in chunks of BLOCK_SIZE
    for chunk in range(0, NUM_CHUNKS):
        idx = chunk * BLOCK_SIZE + offs
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        sum_vec += x_f32
        sum_sq_vec += x_f32 * x_f32

    # Reduce across the BLOCK_SIZE lanes to scalars
    sum_val = tl.sum(sum_vec, axis=0)
    sum_sq_val = tl.sum(sum_sq_vec, axis=0)

    S_f32 = tl.full((), S, dtype=tl.float32)
    mean = sum_val / S_f32
    mean_sq = mean * mean
    var = sum_sq_val / S_f32 - mean_sq
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ------------------------------------------------------------------
    # Pass 2: normalize and write output
    # ------------------------------------------------------------------
    for chunk in range(0, NUM_CHUNKS):
        idx = chunk * BLOCK_SIZE + offs
        mask = idx < S
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        y_f32 = (x_f32 - mean) * inv_std
        y = y_f32.to(x.dtype)

        tl.store(y_ptr + row_start + idx, y, mask=mask)


def triton_instance_norm_2d(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    InstanceNorm2d via a single Triton kernel.
    Normalizes over (H, W) per (N, C) instance, no affine transform.
    Matches nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False).
    """
    assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
    assert x.dim() == 4, "Expected NCHW input."

    N, C, H, W = x.shape
    M = N * C
    S = H * W

    x_flat = x.contiguous().view(M, S)
    y_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = 256  # power-of-2 as required
    NUM_CHUNKS = triton.cdiv(S, BLOCK_SIZE)

    grid = lambda meta: (M,)

    instance_norm_single_pass_kernel[grid](
        x_flat,
        y_flat,
        M,
        S,
        eps,
        NUM_CHUNKS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return y_flat.view(N, C, H, W)


class ModelNew(nn.Module):
    """
    Triton-accelerated Instance Normalization (2D, NCHW) without affine.
    Matches nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False).
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_instance_norm_2d(x, eps=self.eps)
