# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_2d(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 1D grid over (M, N) tiles, with GROUP_M grouping along M to improve L2 reuse of B
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-tile for A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_offsets = k_iter + offs_k
        k_mask = k_offsets < K

        a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def matmul_tall_skinny_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    # 1D program id: each program computes ROWS_PER_PROGRAM rows of C,
    # each of length N (N is small, e.g., <= 32).
    pid = tl.program_id(axis=0)

    offs_m = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator for ROWS_PER_PROGRAM x N
    acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        offs_k = k_iter + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A tile: [ROWS_PER_PROGRAM, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        k_iter += BLOCK_K

    # Store back results
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def _next_power_of_2_up_to_32(x: int) -> int:
    """Return next power of two >= x, clamped to at most 32."""
    if x <= 1:
        return 1
    v = 1 << (x - 1).bit_length()
    return min(32, v)


def _triton_matmul_tall_skinny(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Specialized matmul for tall-skinny output along N (N small, e.g., <= 32).

    A: [M, K]
    B: [K, N]  (N small)
    Returns C: [M, N]
    """
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    BLOCK_N = _next_power_of_2_up_to_32(N)
    BLOCK_K = 64  # tile K; power-of-two
    ROWS_PER_PROGRAM = 4

    grid = (triton.cdiv(M, ROWS_PER_PROGRAM),)

    matmul_tall_skinny_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
        num_warps=4,
        num_stages=2,
    )
    return c


def _triton_matmul_general(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    General high-performance matmul using a grouped 1D grid over (M, N) tiles.
    """
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_kernel_2d[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return c


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton matmul: C = A @ B
    A: [M, K], B: [K, N], returns C: [M, N]

    Shape-aware dispatch:
      - If one output dimension is very small (<= 32), use a GEMV-style
        1D grid where each program computes full N-vector(s) for one/few rows.
      - Otherwise, use a grouped 1D-tiled GEMM for better L2 reuse and SM
        utilization on RTX 4090.
    """
    assert a.ndim == 2 and b.ndim == 2, "Inputs must be 2D matrices"
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible matrix shapes"

    # Case 1: output is tall-skinny with small N (M >> N)
    if N <= 32 and M >= N:
        return _triton_matmul_tall_skinny(a, b)

    # Case 2: output is wide-short with small M (N >> M):
    # Use (A @ B)^T = B^T @ A^T and reuse the tall-skinny kernel.
    if M <= 32 and N > M:
        a_t = b.transpose(0, 1)  # [N, K]
        b_t = a.transpose(0, 1)  # [K, M] (M small)
        c_t = _triton_matmul_tall_skinny(a_t, b_t)  # [N, M]
        return c_t.transpose(0, 1)  # [M, N]

    # General high-performance GEMM
    return _triton_matmul_general(a, b)


class ModelNew(nn.Module):
    """
    Triton-accelerated model that performs a single matrix multiplication (C = A @ B).
    Optimized for general GEMM and tall-skinny cases on RTX 4090.
    """
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)
