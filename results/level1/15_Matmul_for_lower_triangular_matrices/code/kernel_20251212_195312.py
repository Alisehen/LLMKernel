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
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized lower triangular matrix multiplication kernel.
    Uses grouped block processing to reduce shared memory requirements.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group blocks to improve L2 cache hit rate
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Early exit for blocks entirely in upper triangle
    if pid_m * BLOCK_SIZE_M < pid_n * BLOCK_SIZE_N:
        return
    
    # Create offsets for M dimension (rows)
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rmask_m = rm < N
    
    # Create offsets for N dimension (columns)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rmask_n = rn < N
    
    # Only compute blocks where some elements are in lower triangle
    row_min = pid_m * BLOCK_SIZE_M
    col_max = (pid_n + 1) * BLOCK_SIZE_N
    if row_min < col_max:  # Some overlap with lower triangle
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
            
            # Accumulate matrix multiplication
            acc += tl.dot(a, b, allow_tf32=True)
        
        # Store only elements in lower triangle
        c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        store_mask = (rmask_m[:, None] & rmask_n[None, :]) & (rm[:, None] >= rn[None, :])
        tl.store(c_ptrs, acc, mask=store_mask)


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
    
    # Allocate output (initialized to zeros for upper triangle)
    C = torch.zeros_like(A)
    
    # Conservative block sizes to avoid shared memory overflow
    # These values ensure shared memory usage < 101376 bytes
    if N >= 4096:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 4
    elif N >= 2048:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 4
    elif N >= 1024:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 4
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
    
    # Calculate grid size with grouping
    grid_m = triton.cdiv(N, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)
    
    # Launch kernel with appropriate grid
    def grid(META):
        return (triton.cdiv(grid_m, META['GROUP_SIZE_M']) * grid_n * META['GROUP_SIZE_M'],)
    
    tril_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=3,
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
