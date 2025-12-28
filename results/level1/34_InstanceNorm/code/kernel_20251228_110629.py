# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def instance_norm_partial_sums_kernel(
    x_ptr,          # *f32 / *f16 / *bf16, [M, S] flattened
    sum_ptr,        # *f32, [M]
    sum_sq_ptr,     # *f32, [M]
    M,              # int: number of (N, C) rows
    S,              # int: spatial size H*W
    NUM_CHUNKS,     # int: number of chunks along S
    BLOCK_SIZE: tl.constexpr,  # power-of-2, e.g., 256
):
    pid = tl.program_id(axis=0)

    row = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS

    # Each row has S contiguous elements
    row_start = row * S

    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < S

    idx = row_start + offsets
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    sum_val = tl.sum(x_f32, axis=0)
    sum_sq_val = tl.sum(x_f32 * x_f32, axis=0)

    tl.atomic_add(sum_ptr + row, sum_val)
    tl.atomic_add(sum_sq_ptr + row, sum_sq_val)


@triton.jit
def instance_norm_forward_kernel(
    x_ptr,          # *f32 / *f16 / *bf16, [M, S] flattened
    y_ptr,          # same dtype as x, [M, S]
    sum_ptr,        # *f32, [M]
    sum_sq_ptr,     # *f32, [M]
    M,              # int
    S,              # int
    eps,            # float32
    NUM_CHUNKS,     # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    row = pid // NUM_CHUNKS
    chunk = pid % NUM_CHUNKS

    row_start = row * S

    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < S
    idx = row_start + offsets

    # Load row statistics
    sum_val = tl.load(sum_ptr + row)
    sum_sq_val = tl.load(sum_sq_ptr + row)

    S_f32 = tl.full((), S, dtype=tl.float32)
    mean = sum_val / S_f32
    mean_sq = mean * mean
    var = sum_sq_val / S_f32 - mean_sq
    inv_std = 1.0 / tl.sqrt(var + eps)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    y_f32 = (x_f32 - mean) * inv_std
    y = y_f32.to(x.dtype)

    tl.store(y_ptr + idx, y, mask=mask)


def triton_instance_norm_2d(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    InstanceNorm2d via Triton.
    Normalizes over (H, W) per (N, C) instance, no affine transform.
    """
    assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
    assert x.dim() == 4, "Expected NCHW input."

    N, C, H, W = x.shape
    M = N * C
    S = H * W

    x_flat = x.contiguous().view(M, S)
    y_flat = torch.empty_like(x_flat)

    # Accumulators for per-(N,C) statistics
    sum_buf = torch.zeros(M, device=x.device, dtype=torch.float32)
    sum_sq_buf = torch.zeros(M, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 256  # power-of-2 as required
    NUM_CHUNKS = triton.cdiv(S, BLOCK_SIZE)

    grid = lambda meta: (M * NUM_CHUNKS,)

    # Pass 1: compute partial sums and sum of squares
    instance_norm_partial_sums_kernel[grid](
        x_flat,
        sum_buf,
        sum_sq_buf,
        M,
        S,
        NUM_CHUNKS,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2: normalize
    instance_norm_forward_kernel[grid](
        x_flat,
        y_flat,
        sum_buf,
        sum_sq_buf,
        M,
        S,
        eps,
        NUM_CHUNKS,
        BLOCK_SIZE=BLOCK_SIZE,
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
