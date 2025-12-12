import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for computing C = A @ B.
    Optimized for large K dimension (K >> M, N).
    Using simpler grid mapping without grouping to avoid errors.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Decompose program ID
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Block offsets with boundary checking
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Create pointers for A and B blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Main K loop - optimized for large K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        k_size = min(BLOCK_SIZE_K, k_remaining)
        
        if k_size > 0:
            # Load A block with boundary checking
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_size)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            # Load B block with boundary checking  
            b_mask = (offs_k[:, None] < k_size) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # Accumulate dot product
            acc += tl.dot(a, b, allow_tf32=False)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result with boundary checking
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using Triton.
    
    Args:
        a: Tensor of shape (M, K)
        b: Tensor of shape (K, N)
    
    Returns:
        Tensor of shape (M, N)
    """
    # Check dimensions
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, f"Dimension mismatch: K1={K1}, K2={K2}"
    K = K1
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Configuration optimized for large K
    # Reduced block sizes to fit within shared memory limits
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Compute grid - simple 1D grid without grouping
    num_blocks_m = triton.cdiv(M, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(N, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n,)
    
    # Launch kernel with optimized configuration
    # Disable tf32 for numerical stability with large K
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        num_stages=3,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized matrix multiplication model using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using Triton.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return triton_matmul(A, B)
