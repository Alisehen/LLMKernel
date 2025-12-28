# High-performance Triton matmul (replacement for torch.matmul)

import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    # 2D program id: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers to the first K-tile for A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # accumulator in fp32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    k_iter = 0
    while k_iter < K:
        k_offsets = k_iter + offs_k

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs + k_iter * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs + k_iter * stride_bk, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)
        k_iter += BLOCK_K

    # write back
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # let Triton cast acc to the output dtype automatically
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton matmul: C = A @ B
    A: [M, K], B: [K, N], returns C: [M, N]
    Works best when one dimension is tall/skinny (e.g., M >> N or N >> M).
    """
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible matrix shapes"

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Extract strides (row-major assumed but supports non-contiguous via strides)
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    # Choose tiling tuned for tall/skinny shapes:
    # If one dimension is very small, use narrow tiles on that dimension.
    if min(M, N) <= 32:
        BLOCK_M = 128
        BLOCK_N = 32
        BLOCK_K = 32
        num_warps = 4
    else:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        num_warps = 8

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs a single matrix multiplication (C = A @ B).
    Optimized for cases where one matrix dimension is tall and skinny.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)
