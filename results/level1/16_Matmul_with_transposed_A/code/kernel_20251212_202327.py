import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_transposed_kernel(
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
    """Optimized matrix multiplication kernel for A.T @ B with improved occupancy."""
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
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointers with correct strides
    A_ptr = A_ptr + (offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am)
    B_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Loop over K in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K
        mask_k = (k_offs + offs_k) < K
        
        # Load A block with transpose: A[k, m] -> A[m, k]
        a_ptrs = A_ptr + (k_offs * stride_ak)
        a = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        
        # Load B block
        b_ptrs = B_ptr + (k_offs * stride_bk)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Matrix multiplication with tensor cores
        if USE_TF32:
            acc = tl.dot(a, b, acc, allow_tf32=True)
        else:
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, allow_tf32=False)
    
    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized matrix multiplication A.T @ B with autotuning."""
    K, M = A.shape
    K_B, N = B.shape
    assert K == K_B, f"Dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
    
    # Ensure tensors are contiguous and on correct device
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Enable TF32 for better performance on Ada Lovelace
    USE_TF32 = True
    
    # Calculate grid dynamically based on autotuned parameters
    def grid(meta):
        num_pid_m = triton.cdiv(M, meta['BLOCK_SIZE_M'])
        num_pid_n = triton.cdiv(N, meta['BLOCK_SIZE_N'])
        return (num_pid_m * num_pid_n,)
    
    # Launch kernel with autotuned parameters
    matmul_transposed_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        USE_TF32=USE_TF32,
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
        return triton_matmul_transposed(A, B)
