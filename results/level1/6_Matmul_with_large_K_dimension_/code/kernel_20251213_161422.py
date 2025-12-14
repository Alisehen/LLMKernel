# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Original config (must be included)
        triton.Config({}, num_warps=8, num_stages=4),
        # Nearby variants for final micro-tuning
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program id: [BLOCK_M, BLOCK_N] tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers to the first K tile for A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute M/N masks (independent of K)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    k_remaining = K
    while k_remaining > 0:
        # Mask along K for this iteration
        k_mask_a = offs_k[None, :] < k_remaining
        k_mask_b = offs_k[:, None] < k_remaining

        a = tl.load(a_ptrs, mask=m_mask & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & n_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_remaining -= BLOCK_K

    # Write back
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        # Original config (must be included)
        triton.Config({}, num_warps=8, num_stages=4),
        # Nearby variants for final micro-tuning
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=["M", "N", "K", "SPLIT_K"],
)
@triton.jit
def matmul_kernel_splitk(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    SPLIT_K,  # runtime int, number of K-splits
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Split-K matmul: multiple program_ids along K each compute a partial sum
    and accumulate into C via atomic_add.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Compute this program's K range [k_start, k_end)
    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    # Early exit if this split is entirely out of range
    if k_start >= K:
        return

    # Pointers to the first K tile of this split
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    k_iter = k_start
    while k_iter < k_end:
        # Mask elements whose K-index is within this split's [k_start, k_end)
        k_mask_a = (k_iter + offs_k[None, :]) < k_end
        k_mask_b = (k_iter + offs_k[:, None]) < k_end

        a = tl.load(a_ptrs, mask=m_mask & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & n_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Atomic accumulate into C
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_mask & n_mask
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def _triton_matmul_single(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Single-call Triton matmul: C = A @ B
    A: [M, K], B: [K, N], result float32.
    Uses split-K parallelism when grid would otherwise be too small.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    # Strides (row-major or generic)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()

    # Tile sizes tuned for Ada (4090)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Estimate base 2D grid size
    tiles_m = triton.cdiv(M, BLOCK_M)
    tiles_n = triton.cdiv(N, BLOCK_N)
    num_tiles_2d = tiles_m * tiles_n

    # Target minimum number of blocks to keep 128 SMs busy.
    # Rough heuristic: aim for at least 8 blocks per SM.
    TARGET_MIN_BLOCKS = 8 * 128  # 1024

    # Max split-K given K so each split has enough work (>= ~4 BLOCK_K iterations)
    max_split_from_k = max(1, K // (BLOCK_K * 4))

    # Decide whether to use split-K
    if num_tiles_2d >= TARGET_MIN_BLOCKS or max_split_from_k == 1:
        # Enough 2D parallelism; use standard kernel
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return C
    else:
        # Not enough 2D tiles: introduce split-K to increase grid size along K
        desired_blocks = TARGET_MIN_BLOCKS
        # How many K-splits would reach the desired number of blocks?
        needed_splits = math.ceil(desired_blocks / max(num_tiles_2d, 1))
        # Cap split-K by both K and a reasonable upper bound
        SPLIT_K = int(min(32, max_split_from_k, needed_splits))
        SPLIT_K = max(SPLIT_K, 1)

        # For atomic-add we need C initialized to zero
        C = torch.zeros((M, N), device=A.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
            SPLIT_K,
        )

        matmul_kernel_splitk[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            C.stride(0), C.stride(1),
            SPLIT_K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return C


def triton_large_k_matmul(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """
    Matmul with very large K using chunking to improve numerical stability:
        C = A @ B
    Splits K into chunks and accumulates results in PyTorch.
    Uses Triton kernels (with split-K when beneficial) for each chunk.
    """
    if (not A.is_cuda) or (not B.is_cuda):
        return torch.matmul(A, B)

    compute_dtype = torch.float32
    A_ = A.to(compute_dtype)
    B_ = B.to(compute_dtype)

    M, K = A_.shape
    Kb, N = B_.shape
    assert K == Kb

    C_accum = torch.zeros((M, N), device=A.device, dtype=compute_dtype)

    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        A_chunk = A_[:, k_start:k_end]
        B_chunk = B_[k_start:k_end, :]
        C_chunk = _triton_matmul_single(A_chunk, B_chunk)
        C_accum += C_chunk

    out_dtype = torch.result_type(A, B)
    if out_dtype.is_floating_point:
        return C_accum.to(out_dtype)
    else:
        return C_accum.to(out_dtype)


class ModelNew(nn.Module):
    """
    Model using optimized Triton kernels for matrix multiplication with very large K.
    Includes split-K parallelism to improve SM utilization when M/N are small.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_cuda and B.is_cuda:
            return triton_large_k_matmul(A, B)
        else:
            return torch.matmul(A, B)
