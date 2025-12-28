import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline (no split-K)
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1},
            num_stages=3,
            num_warps=4,
        ),
        # Moderate split-K
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8},
            num_stages=4,
            num_warps=4,
        ),
        # More aggressive split-K
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 16},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 8},
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_splitk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K: tl.constexpr,          # K kept constexpr for static loop
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K matmul kernel: each program computes a partial BLOCK_M x BLOCK_N
    tile of C over a shard of the K dimension, then atomically accumulates
    into the global output matrix C.
    """
    # 3D program id: tile over M, N, and split-K
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    # Offsets for this program's tile in M and N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_broadcast = offs_m[:, None]      # (BM, 1)
    offs_n_broadcast = offs_n[None, :]      # (1, BN)

    # Initialize local accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K indices within a BLOCK_K chunk
    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_k, BLOCK_K)

    # Step between iterations in terms of K (each split processes disjoint blocks)
    k_block_stride = BLOCK_K * SPLIT_K

    # Loop over K in a split-K fashion:
    # different pid_k values walk over interleaved BLOCK_K tiles of K.
    for k_outer in range(0, K, k_block_stride):
        # Base K index for this split
        k_base = k_outer + pid_k * BLOCK_K
        k_curr = k_base + offs_k  # (BLOCK_K,)

        # Pointers for A and B tiles
        a_ptrs = A_ptr + offs_m_broadcast * stride_am + k_curr[None, :] * stride_ak  # (BM, BK)
        b_ptrs = B_ptr + k_curr[:, None] * stride_bk + offs_n_broadcast * stride_bn  # (BK, BN)

        # Masks for bounds
        k_mask_row = k_curr[None, :] < K            # (1, BK)
        k_mask_col = k_curr[:, None] < K            # (BK, 1)

        a_mask = (offs_m_broadcast < M) & k_mask_row              # (BM, BK)
        b_mask = k_mask_col & (offs_n_broadcast < N)              # (BK, BN)

        # Load tiles with masking
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Fused block matmul
        acc += tl.dot(a, b, allow_tf32=True)

    # Atomically accumulate partial result into C
    c_ptrs = C_ptr + offs_m_broadcast * stride_cm + offs_n_broadcast * stride_cn
    c_mask = (offs_m_broadcast < M) & (offs_n_broadcast < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton matmul with split-K parallelism: C = A @ B
    Expects A shape (M, K), B shape (K, N), dtype=float32, CUDA tensors.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Expected float32 tensors"
    assert A.shape[1] == B.shape[0], "Incompatible matmul shapes"

    A = A.contiguous()
    B = B.contiguous()

    M, K = A.shape
    K2, N = B.shape
    assert K2 == K

    # C must be zero-initialized because we use atomic_add for split-K reduction
    C = torch.zeros((M, N), device=A.device, dtype=A.dtype)

    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    def grid(meta):
        # 3D launch grid: (M tiles, N tiles, split-K tiles)
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            meta['SPLIT_K'],
        )

    matmul_splitk_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for torch.matmul(A, B) with large K dimension.
    Matches the behavior of the reference Model.forward (C = A @ B).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Move to CUDA if needed for Triton execution
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        return triton_matmul(A, B)
