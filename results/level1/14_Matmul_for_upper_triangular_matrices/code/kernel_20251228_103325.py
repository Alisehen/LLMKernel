import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_triu_kernel(
    A_ptr, B_ptr, C_ptr,
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

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k = 0
    while k < K:
        k_offsets = k + offs_k  # [BLOCK_K]

        # Pointers for current tiles of A and B
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks to guard memory accesses
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate matrix product for this K-slice
        acc += tl.dot(a, b, allow_tf32=True)

        k += BLOCK_K

    # Compute row/col indices for this tile
    row_idx = offs_m[:, None]  # (BLOCK_M, 1)
    col_idx = offs_n[None, :]  # (1, BLOCK_N)

    # Valid in-bounds mask
    in_bounds = (row_idx < M) & (col_idx < N)
    # Upper-triangular mask (including diagonal)
    is_triu = col_idx >= row_idx
    out_mask = in_bounds & is_triu

    # Store only upper-triangular part; lower-triangular remains zero
    c_ptrs = C_ptr + row_idx * stride_cm + col_idx * stride_cn
    tl.store(c_ptrs, acc, mask=out_mask)


def triton_matmul_triu(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute torch.triu(A @ B) using a Triton kernel.

    A: (M, K)
    B: (K, N)
    Returns: (M, N) upper-triangular result.
    """
    assert A.dim() == 2 and B.dim() == 2, "Only 2D matrices are supported"
    assert A.size(1) == B.size(0), "Incompatible matrix dimensions"
    assert A.device == B.device, "A and B must be on the same device"
    assert A.is_cuda, "Triton matmul requires CUDA tensors"

    A_mat = A.contiguous()
    B_mat = B.contiguous()

    M, K = A_mat.shape
    Kb, N = B_mat.shape
    assert K == Kb

    # Initialize C with zeros so lower-triangular part is correct
    C = torch.zeros((M, N), device=A_mat.device, dtype=A_mat.dtype)

    # Strides for row-major layout
    stride_am, stride_ak = A_mat.stride()
    stride_bk, stride_bn = B_mat.stride()
    stride_cm, stride_cn = C.stride()

    # Power-of-2 block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Grid: number of program instances
    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta["BLOCK_M"])),
            max(1, triton.cdiv(N, meta["BLOCK_N"])),
        )

    matmul_triu_kernel[grid](
        A_mat,
        B_mat,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )

    return C


class ModelNew(nn.Module):
    """
    Triton-accelerated model computing C = triu(A @ B).
    """

    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul_triu(A, B)
