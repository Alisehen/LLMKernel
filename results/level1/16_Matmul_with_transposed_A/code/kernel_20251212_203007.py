import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transposed_kernel_optimized(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_ak,
    stride_am,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TF32: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Optimized matrix multiplication kernel for A.T @ B with improved memory access patterns."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets for blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checks
    mask_m = rm < M
    mask_n = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Precompute loop bounds
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Main loop with improved memory access patterns
    for k in range(0, k_tiles):
        k_offs = k * BLOCK_SIZE_K
        mask_k = (k_offs + rk) < K
        
        # Load A block (transposed access)
        a_ptrs = A_ptr + ((k_offs + rk)[None, :] * stride_ak + rm[:, None] * stride_am)
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block
        b_ptrs = B_ptr + ((k_offs + rk)[:, None] * stride_bk + rn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiplication
        if USE_TF32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b, allow_tf32=False)
    
    # Store result
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_transposed_optimized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized matrix multiplication A.T @ B with improved parameters."""
    M, K = A.shape[1], A.shape[0]
    N = B.shape[1]
    
    # Ensure tensors are contiguous and on correct device
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Optimized block sizes that fit within shared memory constraints
    # Reduced from 128/128/64 to fit within 101376 shared memory limit
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 4
    
    # Enable TF32 for better performance on supported hardware
    USE_TF32 = True
    
    # Calculate grid
    num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_groups = triton.cdiv(num_pid_m, GROUP_SIZE_M)
    grid = (num_groups * num_pid_in_group,)
    
    # Launch kernel with optimized parameters
    # Reduced num_stages from 4 to 3 to fit within shared memory
    # Keep num_warps at 8 for good performance
    matmul_transposed_kernel_optimized[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TF32=USE_TF32,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=8,
        num_stages=3,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model with Triton kernels for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication A.T @ B using Triton kernels.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transposed_optimized(A, B)
