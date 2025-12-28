import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=5,
            num_warps=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K: tl.constexpr,         # K must be constexpr for the static range loop
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program id: each program computes one BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
    offs_m_broadcast = offs_m[:, None]                # (BLOCK_M, 1)
    offs_n_broadcast = offs_n[None, :]                # (1, BLOCK_N)

    # Initialize accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop: static range because K is tl.constexpr
    offs_k = tl.arange(0, BLOCK_K)
    for k in range(0, K, BLOCK_K):
        k_curr = k + offs_k

        # Pointers for A and B tiles
        a_ptrs = A_ptr + offs_m_broadcast * stride_am + k_curr[None, :] * stride_ak  # (BM, BK)
        b_ptrs = B_ptr + k_curr[:, None] * stride_bk + offs_n_broadcast * stride_bn  # (BK, BN)

        # Masks for out-of-bounds
        a_mask = (offs_m_broadcast < M) & (k_curr[None, :] < K)    # (BM, BK)
        b_mask = (k_curr[:, None] < K) & (offs_n_broadcast < N)    # (BK, BN)

        # Safe loads with masking
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Block dot-product; allow TF32 for speed on Ampere+
        acc += tl.dot(a, b, allow_tf32=True)

    # Write back result
    c_ptrs = C_ptr + offs_m_broadcast * stride_cm + offs_n_broadcast * stride_cn
    c_mask = (offs_m_broadcast < M) & (offs_n_broadcast < N)       # (BM, BN)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton matmul: C = A @ B
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

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bk = B.stride(0)
    stride_bn = B.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    def grid(meta):
        # 2D launch grid: one program per C tile
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_kernel[grid](
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
