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
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_warps=8,
            num_stages=5,
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
    """Optimized lower triangular matrix multiplication with improved memory patterns."""
    # 2D block IDs for better parallelism
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Skip upper triangle blocks
    if pid_m < pid_n:
        return
    
    # Row offsets for A and C
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask_m = rm < N
    
    # Column offsets for B and C
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rmask_n = rn < N
    
    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pre-compute triangular mask for the block (outside K-loop)
    tri_mask = rm[:, None] >= rn[None, :]
    
    # Pointer arithmetic optimized for coalesced access
    a_ptrs_base = a_ptr + (rm[:, None] * stride_am)
    b_ptrs_base = b_ptr + (rn[None, :] * stride_bn)
    
    # Main K-loop with prefetching and vectorized loads
    for k in range(0, N, BLOCK_SIZE_K):
        k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < N
        
        # Load A block with triangular condition
        a_ptrs = a_ptrs_base + k_offsets[None, :] * stride_ak
        a_mask = (rmask_m[:, None] & mask_k[None, :]) & (rm[:, None] >= k_offsets[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with triangular condition
        b_ptrs = b_ptrs_base + k_offsets[:, None] * stride_bk
        b_mask = (mask_k[:, None] & rmask_n[None, :]) & (k_offsets[:, None] >= rn[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiplication with tensor cores
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Store only lower triangular elements
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    store_mask = (rmask_m[:, None] & rmask_n[None, :]) & tri_mask
    tl.store(c_ptrs, acc, mask=store_mask)


def tril_matmul_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for triangular matrix multiplication with optimized 2D grid."""
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"
    assert A.shape == B.shape, "Matrices must have same shape"
    N = A.size(0)
    
    # Allocate output (initialized to zeros)
    C = torch.zeros_like(A)
    
    # Calculate 2D grid for better parallelism
    def grid(META):
        BLOCK_SIZE_M = META["BLOCK_SIZE_M"]
        BLOCK_SIZE_N = META["BLOCK_SIZE_N"]
        grid_m = triton.cdiv(N, BLOCK_SIZE_M)
        grid_n = triton.cdiv(N, BLOCK_SIZE_N)
        return (grid_m, grid_n)
    
    # Launch kernel with optimized parameters
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
