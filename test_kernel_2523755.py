import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel_optimized(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel with reduced shared memory usage.
    """
    # Program ID management
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(axis=1)
    
    # Number of program IDs along M and N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Group ID and size for better cache utilization
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Program IDs within the group
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Number of K tiles to process
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Loop over K dimension
    for k in range(0, k_tiles):
        # Compute K offset for this tile
        k_offset = k * BLOCK_SIZE_K
        k_indices = k_offset + offs_k
        
        # Load A block with proper masking
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + k_indices[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M) & (k_indices[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block with proper masking
        b_ptrs = b_ptr + (k_indices[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b_mask = (k_indices[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply with TF32 precision
        accumulator += tl.dot(a, b, allow_tf32=True)
    
    # Write back result with masking
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for optimized Triton matrix multiplication.
    """
    # Check constraints
    assert a.dim() == 2, "Matrix A must be 2D"
    assert b.dim() == 2, "Matrix B must be 2D"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Heuristic for block sizes with reduced shared memory usage
    # Based on matrix dimensions and shared memory limits
    if M <= 64 and N <= 64:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
        GROUP_SIZE_M = 8
        SPLIT_K = 1
        num_warps = 4
        num_stages = 3
    elif M <= 128 and N <= 128:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 32
        GROUP_SIZE_M = 8
        SPLIT_K = 1
        num_warps = 4
        num_stages = 3
    else:
        # Conservative settings for large matrices to avoid shared memory overflow
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 32
        GROUP_SIZE_M = 8
        SPLIT_K = 1
        num_warps = 4
        num_stages = 2  # Reduced from 4 to reduce shared memory usage
    
    # Ensure block sizes are powers of two
    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()
    
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, next_power_of_two(M))
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, next_power_of_two(N))
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, next_power_of_two(K))
    
    # Compute grid
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        META['SPLIT_K'],
    )
    
    # Launch kernel
    matmul_kernel_optimized[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)
