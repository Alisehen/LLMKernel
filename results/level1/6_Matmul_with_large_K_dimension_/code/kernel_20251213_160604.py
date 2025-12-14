# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# High-performance matmul kernel: C = A @ B
# A: [M, K], row-major
# B: [K, N], row-major
# C: [M, N], row-major
# ------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
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
    # Tile indices
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for M and N dimensions
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    # Dynamically loop over K to avoid massive unrolling for very large K
    while k < K:
        k_offsets = k + offs_k

        # Pointers for this K-tile
        a_ptrs = a_ptr + (
            offs_m[:, None] * stride_am +
            k_offsets[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            k_offsets[:, None] * stride_bk +
            offs_n[None, :] * stride_bn
        )

        # K-bound mask
        k_mask = k_offsets < K

        # Load A and B with proper masks
        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # FMA on the K-tile
        acc += tl.dot(a, b)

        k += BLOCK_K

    # Write back C
    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm +
        offs_n[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ------------------------------------------------------------
# Wrapper: launch Triton matmul kernel
# ------------------------------------------------------------
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using Triton.
    A: [M, K], B: [K, N]
    """
    assert A.is_cuda and B.is_cuda
    assert A.shape[1] == B.shape[0]

    # Promote to float32 for stable accumulation
    compute_dtype = torch.float32
    A_ = A.to(compute_dtype)
    B_ = B.to(compute_dtype)

    M, K = A_.shape
    K2, N = B_.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=compute_dtype)

    stride_am, stride_ak = A_.stride()
    stride_bk, stride_bn = B_.stride()
    stride_cm, stride_cn = C.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    matmul_kernel[grid](
        A_, B_, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    # Cast back to result type
    out_dtype = torch.result_type(A, B)
    return C.to(out_dtype)


# ------------------------------------------------------------
# nn.Module wrapper
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model using optimized Triton kernels for matrix multiplication:
        C = A @ B
    Handles very large K via a dynamically-looped kernel to avoid
    excessive unrolling while keeping correctness and high performance.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if A.is_cuda and B.is_cuda:
            return triton_matmul(A, B)
        else:
            return torch.matmul(A, B)
