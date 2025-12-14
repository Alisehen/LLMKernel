import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Optimized for Ada Lovelace: higher M dimension for better warp occupancy
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'VEC_SIZE': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'VEC_SIZE': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'VEC_SIZE': 4}, num_warps=8, num_stages=4),
        # Smaller configs for edge cases
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'VEC_SIZE': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'VEC_SIZE': 4}, num_warps=2, num_stages=4),
    ],
    key=['n', 'm'],
)
@triton.jit
def diag_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n, m,
    stride_a,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Optimized diagonal matrix multiplication kernel with:
    - Reordered grid layout for better warp occupancy
    - Efficient vectorization and tiling
    - Optimized memory access patterns
    """
    # Reorder grid: pid_n for columns, pid_m for rows
    # This improves coalescing and occupancy on Ada Lovelace
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Row offsets - using BLOCK_SIZE_M for rows
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < n
    
    # Load diagonal elements once per row block
    a = tl.load(a_ptr + row_offsets * stride_a, mask=row_mask, other=0.0)
    a = a.to(b_ptr.dtype.element_ty)  # Ensure type matching for multiplication
    
    # Column offsets with vectorization - using BLOCK_SIZE_N for columns
    col_start = pid_n * BLOCK_SIZE_N * VEC_SIZE
    col_offsets_base = col_start + tl.arange(0, BLOCK_SIZE_N) * VEC_SIZE
    
    # Create broadcasted indices for efficient 2D access
    row_idx = tl.reshape(row_offsets, (BLOCK_SIZE_M, 1, 1))
    col_offsets_3d = tl.reshape(col_offsets_base, (1, BLOCK_SIZE_N, 1)) + tl.arange(0, VEC_SIZE)
    
    # Load B tile with efficient masking
    b_ptr_base = b_ptr + row_idx * stride_b0
    b_ptr_full = b_ptr_base + col_offsets_3d * stride_b1
    
    # Create combined mask - optimized for 3D access
    row_mask_3d = tl.reshape(row_mask, (BLOCK_SIZE_M, 1, 1))
    col_mask = col_offsets_3d < m
    b_mask = row_mask_3d & col_mask
    
    # Load B with appropriate type
    b = tl.load(b_ptr_full, mask=b_mask, other=0.0)
    
    # Compute C = diag(A) * B (element-wise multiplication)
    a_expanded = tl.reshape(a, (BLOCK_SIZE_M, 1, 1))
    c = a_expanded * b
    
    # Store C with vectorized stores
    c_ptr_base = c_ptr + row_idx * stride_c0
    c_ptr_full = c_ptr_base + col_offsets_3d * stride_c1
    tl.store(c_ptr_full, c, mask=b_mask)

def triton_diag_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes C = diag(A) @ B using optimized Triton kernel.
    
    Args:
        A: 1D tensor of shape (N,)
        B: 2D tensor of shape (N, M)
        
    Returns:
        C: 2D tensor of shape (N, M)
    """
    assert A.dim() == 1 and B.dim() == 2, "A must be 1D and B must be 2D"
    N, M = B.shape
    assert A.shape[0] == N, "A must have same length as B's rows"
    
    C = torch.empty_like(B)
    
    # Calculate optimal grid size for better SM utilization
    # 4090 has 128 SMs, target 256-512 blocks minimum
    # Reordered grid: columns first for better coalescing
    grid_m = triton.cdiv(N, 128)  # Start with conservative row block size
    grid_n = triton.cdiv(M, 128)  # Will be auto-tuned by triton
    
    # Ensure minimum grid size for good utilization
    total_blocks = grid_m * grid_n
    if total_blocks < 256:
        # Adjust to ensure at least 256 blocks for occupancy
        if N < 256:
            grid_m = max(2, triton.cdiv(N, 64))
        if M < 256:
            grid_n = max(2, triton.cdiv(M, 64))
    
    grid = (grid_n, grid_m)  # Note: (columns, rows) for better coalescing
    
    # Launch autotuned kernel
    diag_matmul_kernel[grid](
        A, B, C,
        N, M,
        A.stride(0),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication of a diagonal matrix
    with another matrix using high-performance Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication C = diag(A) @ B.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal. Shape: (N,).
            B (torch.Tensor): A 2D tensor. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return triton_diag_matmul(A, B)
