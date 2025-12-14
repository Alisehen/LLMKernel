import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 16},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 16},
            num_warps=8,
            num_stages=4,
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
    """Optimized lower triangular matrix multiplication with improved occupancy."""
    # 1D block ID
    pid = tl.program_id(0)
    
    # Convert linear ID to triangular tile coordinates (i,j) where i>=j
    # Using floating point sqrt for O(1) mapping
    pid_f = tl.cast(pid, tl.float32)
    i = tl.cast(tl.floor((tl.sqrt(8.0 * pid_f + 1.0) - 1.0) / 2.0), tl.int32)
    j = pid - i * (i + 1) // 2
    
    pid_m = i
    pid_n = j
    
    # Row offsets for A and C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask_m = rm < N
    
    # Column offsets for B and C
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rmask_n = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointer increments for better memory coalescing
    a_ptr_base = a_ptr + (rm[:, None] * stride_am)
    b_ptr_base = b_ptr + (rn[None, :] * stride_bn)
    
    # Precompute triangular mask once (outside K-loop) for better performance
    tri_mask_mn = rm[:, None] >= rn[None, :]
    
    # Loop over K dimension with tensor cores
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < N
        
        # Load block from A: A[rm, kk] with triangular mask
        a_ptrs = a_ptr_base + k_offsets[None, :] * stride_ak
        a_mask = (rmask_m[:, None] & mask_k[None, :]) & (rm[:, None] >= k_offsets[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load block from B: B[kk, rn] with triangular mask  
        b_ptrs = b_ptr_base + k_offsets[:, None] * stride_bk
        b_mask = (mask_k[:, None] & rmask_n[None, :]) & (k_offsets[:, None] >= rn[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate using tensor cores (TF32 for better performance)
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store only lower triangular elements
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    store_mask = (rmask_m[:, None] & rmask_n[None, :]) & tri_mask_mn
    tl.store(c_ptrs, acc, mask=store_mask)


def tril_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for triangular matrix multiplication with optimized grid layout."""
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"
    assert A.shape == B.shape, "Matrices must have same shape"
    N = A.size(0)
    
    # Allocate output (initialized to zeros for upper triangle)
    C = torch.zeros_like(A)
    
    # Calculate grid size based on block dimensions
    def grid(META):
        BLOCK_SIZE_M = META["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = META["BLOCK_SIZE_N"]
        grid_m = triton.cdiv(N, BLOCK_SIZE_M)
        # Total number of triangular tiles
        total_tiles = grid_m * (grid_m + 1) // 2
        return (total_tiles,)
    
    # Launch kernel with minimal parameters
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C


class ModelNew(nn.Module):
    """Optimized model for triangular matrix multiplication using Triton."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """Performs matrix multiplication of lower triangular matrices."""
        return tril_matmul_triton(A, B)
