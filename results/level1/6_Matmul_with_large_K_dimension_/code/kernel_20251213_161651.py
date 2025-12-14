# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=4),
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
    # Program IDs for 2D tiling of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-tile for A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Masks for M and N (independent of K)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Iterate over K dimension
    k = 0
    while k < K:
        # Mask for valid K indices in this BLOCK_K slice
        k_mask = k + offs_k < K  # shape: [BLOCK_K]

        a = tl.load(
            a_ptrs,
            mask=m_mask & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask,
            other=0.0,
        )

        acc += tl.dot(a, b)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Write back C tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def _triton_matmul_single(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Single-call Triton matmul: C = A @ B
    A: [M, K], B: [K, N], result float32.
    """
    assert A.is_cuda and B.is_cuda
    assert A.ndim == 2 and B.ndim == 2

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    # Ensure compute in float32
    A_ = A.to(torch.float32)
    B_ = B.to(torch.float32)

    stride_am, stride_ak = A_.stride()
    stride_bk, stride_bn = B_.stride()

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A_, B_, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def triton_large_k_matmul(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """
    Matmul for very large K using chunking along K:
        C = A @ B
    Splits K into chunks and accumulates results in PyTorch to keep kernels shorter.
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

    # Chunk along K and accumulate
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
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_cuda and B.is_cuda:
            return triton_large_k_matmul(A, B)
        else:
            return torch.matmul(A, B)
