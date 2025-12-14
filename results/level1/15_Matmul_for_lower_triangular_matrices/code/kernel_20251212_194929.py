import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def tril_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """
    Lower triangular matrix multiplication kernel optimized for tril(A) * tril(B)
    Only computes elements where row >= col, skipping upper triangular computations
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M dimension
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask_m = rm < N
    
    # Create offsets for N dimension
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rmask_n = rn < N
    
    # Only compute where row >= col (lower triangular condition)
    row_ge_col = rm[:, None] >= rn[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Precompute base pointers for A and B
    a_base = a_ptr + rm[:, None] * stride_am
    b_base = b_ptr + rn[None, :] * stride_bn
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < N
        
        # Load block from A: A[rm, kk], only load lower triangular elements
        a_ptrs = a_base + k_offsets[None, :] * stride_ak
        a_mask = rmask_m[:, None] & mask_k[None, :] & (rm[:, None] >= k_offsets[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load block from B: B[kk, rn], only load lower triangular elements
        b_ptrs = b_base + k_offsets[:, None] * stride_bk
        b_mask = rmask_n[None, :] & mask_k[:, None] & (k_offsets[:, None] >= rn[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store only lower triangular elements
    c_base = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = rmask_m[:, None] & rmask_n[None, :] & row_ge_col
    tl.store(c_base, acc, mask=c_mask)


@triton.jit
def fast_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    High-performance matrix multiplication kernel optimized for triangular matrices
    Uses optimized blocking strategy with tensor cores
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for rows and columns
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks for boundaries
    rmask_m = rm < N
    rmask_n = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Block pointers for A and B
    a_base = a_ptr + rm[:, None] * stride_am
    b_base = b_ptr + rn[None, :] * stride_bn
    
    # Main computation loop with aggressive blocking for L2 reuse
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < N
        
        # Load A block with mask - only load lower triangular elements
        a_ptrs = a_base + k_offsets[None, :] * stride_ak
        a_mask = rmask_m[:, None] & mask_k[None, :] & (rm[:, None] >= k_offsets[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with mask - only load lower triangular elements  
        b_ptrs = b_base + k_offsets[:, None] * stride_bk
        b_mask = rmask_n[None, :] & mask_k[:, None] & (k_offsets[:, None] >= rn[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Use tensor core for accumulation (allow_tf32=True enables tensor cores on Ampere+)
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store result - only store lower triangular
    c_base = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    store_mask = rmask_m[:, None] & rmask_n[None, :] & (rm[:, None] >= rn[None, :])
    tl.store(c_base, acc, mask=store_mask)


def tril_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for triangular matrix multiplication.
    
    Args:
        A: Lower triangular matrix of shape (N, N)
        B: Lower triangular matrix of shape (N, N)
    
    Returns:
        C: Result of A * B with only lower triangular elements computed
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"
    assert A.shape == B.shape, "Matrices must have same shape"
    N = A.size(0)
    
    # Allocate output
    C = torch.empty_like(A)
    
    # Choose optimal block sizes based on matrix size
    if N >= 2048:
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
    elif N >= 1024:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Use fast kernel for large matrices, tril-specific kernel for smaller ones
    if N >= 2048:
        fast_matmul_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    else:
        tril_matmul_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            EVEN_K=N % BLOCK_SIZE_K == 0,
        )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of lower triangular matrices
    using Triton kernels for maximum performance.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.
        
        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).
        
        Returns:
            torch.Tensor: The result of matrix multiplication C = A * B (lower triangular).
        """
        # Use Triton kernel for optimal performance on GPU
        return tril_matmul_triton(A, B)
