import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def matvec_kernel_fp16(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Create masks for valid rows and columns
    m_mask = offs_m < M
    k_mask = offs_k < K
    
    # Pointers to A and B
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k * stride_bk
    
    # Load A and B with appropriate masking, convert to float32 for accumulation
    a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :]).to(tl.float32)
    b = tl.load(B_ptrs, mask=k_mask).to(tl.float32)
    
    # Compute partial dot products
    partial_acc = tl.sum(a * b[None, :], axis=1)
    
    # Use atomic addition to accumulate results across K dimension
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.atomic_add(C_ptrs, partial_acc, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def matvec_kernel_fp32(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # Create masks for valid rows and columns
    m_mask = offs_m < M
    k_mask = offs_k < K
    
    # Pointers to A and B
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k * stride_bk
    
    # Load A and B with appropriate masking
    a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
    b = tl.load(B_ptrs, mask=k_mask)
    
    # Compute partial dot products
    partial_acc = tl.sum(a * b[None, :], axis=1)
    
    # Use atomic addition to accumulate results across K dimension
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.atomic_add(C_ptrs, partial_acc, mask=m_mask)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert B.shape == (K, 1), f"B must be shape ({K}, 1), got {B.shape}"
    
    # Allocate output tensor in float32 for precise accumulation
    C = torch.zeros((M, 1), device=A.device, dtype=torch.float32)
    
    # Choose kernel based on data type
    if A.dtype in [torch.float16, torch.bfloat16]:
        kernel = matvec_kernel_fp16
    else:
        kernel = matvec_kernel_fp32
    
    # Configure grid - using 2D grid for parallelism across rows and K dimension
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(K, META['BLOCK_K']),
    )
    
    # Launch kernel
    kernel[grid](
        A,
        B,
        C,
        M,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        C.stride(0),
    )
    
    # Convert back to original dtype
    return C.to(A.dtype)


class ModelNew(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernels.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)
