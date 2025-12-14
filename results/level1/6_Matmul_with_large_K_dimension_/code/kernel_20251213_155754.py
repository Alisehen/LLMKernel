import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


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
    # Program IDs for 2D grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers to the first K-block of A and B
    # Shapes:
    #   A: [BLOCK_M, BLOCK_K]
    #   B: [BLOCK_K, BLOCK_N]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        # Mask to guard against out-of-bounds in K, M, N
        k_mask_a = offs_k[None, :] + k_iter < K
        k_mask_b = offs_k[:, None] + k_iter < K
        a_mask = (offs_m[:, None] < M) & k_mask_a
        b_mask = k_mask_b & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        # Advance pointers along K
        k_iter += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_matmul_single(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Single-call Triton matmul: C = A @ B
    A: [M, K], B: [K, N], result float32.
    Assumes A, B on CUDA and float32.
    """
    assert A.is_cuda and B.is_cuda
    assert A.dtype == torch.float32 and B.dtype == torch.float32

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    # Strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    # Tile sizes (powers of 2, shared memory safe)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )
    return C


def triton_large_k_matmul(A: torch.Tensor, B: torch.Tensor, chunk_size: int = 65536) -> torch.Tensor:
    """
    Matmul with very large K using chunking to improve numerical stability:
        C = A @ B
    Splits K into chunks and accumulates results in PyTorch.
    """
    # Fallback for unsupported cases
    if (not A.is_cuda) or (not B.is_cuda):
        return torch.matmul(A, B)

    # Promote to float32 for computation
    compute_dtype = torch.float32
    A_ = A.to(compute_dtype)
    B_ = B.to(compute_dtype)

    M, K = A_.shape
    Kb, N = B_.shape
    assert K == Kb

    # Output accumulates in float32
    C_accum = torch.zeros((M, N), device=A.device, dtype=compute_dtype)

    # Ensure chunk_size is >= BLOCK_K and power-of-2 multiple of BLOCK_K is not required,
    # but we choose 65536 (2^16) by default.
    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        A_chunk = A_[:, k_start:k_end]
        B_chunk = B_[k_start:k_end, :]
        C_chunk = _triton_matmul_single(A_chunk, B_chunk)
        C_accum += C_chunk

    # Cast back to result type following PyTorch's promotion rules
    out_dtype = torch.result_type(A, B)
    if out_dtype.is_floating_point:
        return C_accum.to(out_dtype)
    else:
        # For non-floating outputs, let PyTorch handle casting semantics
        return C_accum.to(out_dtype)


class ModelNew(nn.Module):
    """
    Model using a Triton kernel for matrix multiplication with very large K.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Use Triton implementation when possible, otherwise fall back
        if A.is_cuda and B.is_cuda:
            return triton_large_k_matmul(A, B)
        else:
            return torch.matmul(A, B)
