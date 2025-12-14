import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N"],
)
@triton.jit
def tril_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    grid_m,
    grid_n,
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
    Optimized lower triangular matrix multiplication with 1D grid mapping.
    Maps linear block ID to lower triangular tile coordinates.
    """
    # Linear block ID
    pid = tl.program_id(0)
    
    # Convert linear ID to triangular tile coordinates (i,j) where i>=j
    # Using integer arithmetic to avoid floating point in kernel
    # Find largest i such that i*(i+1)/2 <= pid
    i = 0
    total = 0
    while total + i + 1 <= pid:
        i += 1
        total += i
    j = pid - total
    
    pid_m = i
    pid_n = j
    
    # Create offsets for M dimension (rows)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask_m = rm < N
    
    # Create offsets for N dimension (columns)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rmask_n = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < N
        
        # Load block from A: A[rm, kk] with triangular mask
        a_ptrs = a_ptr + (rm[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        a_mask = (rmask_m[:, None] & mask_k[None, :]) & (rm[:, None] >= k_offsets[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load block from B: B[kk, rn] with triangular mask
        b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_mask = (mask_k[:, None] & rmask_n[None, :]) & (k_offsets[:, None] >= rn[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix multiplication using tensor cores
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store only elements in lower triangle
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    store_mask = (rmask_m[:, None] & rmask_n[None, :]) & (rm[:, None] >= rn[None, :])
    tl.store(c_ptrs, acc, mask=store_mask)


def tril_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for triangular matrix multiplication with optimized grid layout.
    
    Args:
        A: Lower triangular matrix of shape (N, N)
        B: Lower triangular matrix of shape (N, N)
    
    Returns:
        C: Result of A * B with only lower triangular elements computed
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"
    assert A.shape == B.shape, "Matrices must have same shape"
    N = A.size(0)
    
    # Allocate output (initialized to zeros for upper triangle)
    C = torch.zeros_like(A)
    
    # Calculate number of blocks in each dimension
    def grid(META):
        BLOCK_SIZE_M = META["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = META["BLOCK_SIZE_N"]
        grid_m = triton.cdiv(N, BLOCK_SIZE_M)
        grid_n = triton.cdiv(N, BLOCK_SIZE_N)
        # Total number of triangular tiles
        total_tiles = grid_m * (grid_m + 1) // 2
        return (total_tiles,)
    
    # Launch kernel with optimized grid mapping
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        triton.cdiv(N, 64),  # grid_m - approximate for kernel
        triton.cdiv(N, 64),  # grid_n - approximate for kernel
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
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
        # Use optimized Triton kernel with autotuning
        return tril_matmul_triton(A, B)
