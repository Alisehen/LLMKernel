import torch
import torch.nn as nn
import triton
import triton.language as tl


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
):
    """Optimized matrix multiplication kernel for A.T @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checks
    mask_m = rm < M
    mask_n = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K in blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Calculate offsets for K dimension
        k_offs = k * BLOCK_SIZE_K
        mask_k = (k_offs + rk) < K
        
        # Load A block with transpose: A[k, m] -> A[m, k]
        a_ptrs = A_ptr + ((k_offs + rk[:, None]) * stride_ak + rm[None, :] * stride_am)
        a = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :])
        
        # Load B block
        b_ptrs = B_ptr + ((k_offs + rk[:, None]) * stride_bk + rn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :])
        
        # Matrix multiplication with optional TF32
        if USE_TF32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a.to(tl.float32), b.to(tl.float32), allow_tf32=False)
    
    # Store result
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized matrix multiplication A.T @ B."""
    M, K = A.shape[1], A.shape[0]
    N = B.shape[1]
    
    # Ensure tensors are contiguous and on correct device
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Choose optimal block sizes based on problem size
    # Use smaller block sizes to fit within shared memory limits
    if M * N * K > 256 * 1024 * 1024:  # Very large problem
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    elif M * N * K > 128 * 1024 * 1024:  # Large problem
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    else:  # Small to medium problem
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    
    # Enable TF32 for better performance on supported hardware
    USE_TF32 = True
    
    # Calculate 2D grid
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    matmul_transposed_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
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
