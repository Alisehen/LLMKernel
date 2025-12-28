import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def upper_tri_matmul_kernel_generic(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Generic 2D tiling kernel (fallback for non-square / non-symmetric shapes)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_offsets = k + offs_k

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tri_mask = offs_m[:, None] <= offs_n[None, :]
    acc = tl.where(tri_mask, acc, 0.0)

    tl.store(c_ptrs, acc, mask=mask_mn)


@triton.jit
def upper_tri_matmul_kernel_optimized(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_tiles_per_dim,  # T = ceil_div(N, BLOCK_M) == ceil_div(M, BLOCK_M)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1D program id over upper-triangular tiles
    pid = tl.program_id(axis=0)

    T = num_tiles_per_dim  # number of tiles along each dimension (runtime scalar)

    # Map 1D pid in [0, T*(T+1)/2) -> (pid_m, pid_n) with 0 <= pid_m <= pid_n < T
    pid_f = tl.astype(pid, tl.float32)
    T_f = tl.astype(T, tl.float32)

    twoT_plus_1 = 2.0 * T_f + 1.0
    disc = twoT_plus_1 * twoT_plus_1 - 8.0 * pid_f
    sqrt_disc = tl.sqrt(disc)
    pid_m_f = (twoT_plus_1 - sqrt_disc) * 0.5
    pid_m = tl.astype(tl.floor(pid_m_f), tl.int32)

    pid_m_f = tl.astype(pid_m, tl.float32)
    start_i_f = pid_m_f * (2.0 * T_f - pid_m_f + 1.0) * 0.5
    start_i = tl.astype(tl.floor(start_i_f), tl.int32)

    pid_n = pid_m + (pid - start_i)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Tile-level K-range exploiting triangular structure:
    # For rows in [row_start, row_start+BLOCK_M) and cols in [col_start, col_start+BLOCK_N),
    # any potentially non-zero contribution must have k in [row_start, col_start+BLOCK_N).
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    k_start = row_start
    # restrict upper bound both by K and by tile column extent
    k_end = tl.minimum(col_start + BLOCK_N, K)

    # Prepare pointers for the first K tile (starting at k_start)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak)
    b_ptrs = B_ptr + ((k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = k_start
    while k < k_end:
        k_offsets = k + offs_k

        # In-bounds masks
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        # Exploit structural zeros of upper-triangular A and B:
        # A[i, k] non-zero only if i <= k
        a_mask = a_mask & (offs_m[:, None] <= k_offsets[None, :])
        # B[k, j] non-zero only if k <= j
        b_mask = b_mask & (k_offsets[:, None] <= offs_n[None, :])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tri_mask = offs_m[:, None] <= offs_n[None, :]
    acc = tl.where(tri_mask, acc, 0.0)

    tl.store(c_ptrs, acc, mask=mask_mn)


def triton_upper_triangular_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes torch.triu(A @ B) using a high-performance Triton kernel.

    A and B are expected to be upper-triangular square matrices (N, N).
    """
    assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D matrices"
    assert A.shape[1] == B.shape[0], "Incompatible matrix shapes"

    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    A_contig = A.contiguous()
    B_contig = B.contiguous()

    M, K = A_contig.shape
    K2, N = B_contig.shape
    assert K == K2, "Inner dimensions must match"

    # Output tensor initialized to zeros so tiles not written by the kernel
    # (strictly lower triangular tiles) remain zero.
    C = torch.zeros((M, N), device=A_contig.device, dtype=A_contig.dtype)

    stride_am, stride_ak = A_contig.stride()
    stride_bk, stride_bn = B_contig.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # If the matrix is square and tiling along M and N matches, use optimized
    # upper-triangular tiling; otherwise, fall back to generic kernel.
    tiles_m = triton.cdiv(M, BLOCK_M)
    tiles_n = triton.cdiv(N, BLOCK_N)

    if (M == N == K) and (tiles_m == tiles_n):
        T = tiles_m
        num_programs = T * (T + 1) // 2
        grid = (num_programs,)

        upper_tri_matmul_kernel_optimized[grid](
            A_contig, B_contig, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            T,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=3,
        )
    else:
        grid = (tiles_m, tiles_n)
        upper_tri_matmul_kernel_generic[grid](
            A_contig, B_contig, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=3,
        )

    return C


class ModelNew(nn.Module):
    """
    Triton-accelerated model that computes the upper triangular part of A @ B.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_upper_triangular_matmul(A, B)
