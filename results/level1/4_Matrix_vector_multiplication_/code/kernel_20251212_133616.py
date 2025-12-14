import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matvec_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (M // BLOCK_M,)
    pid_m = tl.program_id(0)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to A and B
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k * stride_bk
    
    # Accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Main loop with vectorized loads
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Create mask for valid loads
        k_mask = (k * BLOCK_K + offs_k) < K
        m_mask = offs_m < M
        
        # Load A and B with appropriate masking
        a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
        b = tl.load(B_ptrs, mask=k_mask)
        
        # FMA: acc += A * B
        b_broadcast = tl.broadcast_to(b, (BLOCK_M, BLOCK_K))
        acc += tl.sum(a * b_broadcast, axis=1)
        
        # Update pointers for next iteration
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    # Store result
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.store(C_ptrs, acc, mask=m_mask)


@triton.jit
def matvec_kernel_fp16(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Same as matvec_kernel but optimized for FP16/BF16
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k * stride_bk
    
    # Use FP32 accumulator for better precision
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        m_mask = offs_m < M
        
        # Load in FP16/BF16, compute in FP32
        a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :]).to(tl.float32)
        b = tl.load(B_ptrs, mask=k_mask).to(tl.float32)
        
        b_broadcast = tl.broadcast_to(b, (BLOCK_M, BLOCK_K))
        acc += tl.sum(a * b_broadcast, axis=1)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.store(C_ptrs, acc, mask=m_mask)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert B.shape == (K, 1), f"B must be shape ({K}, 1), got {B.shape}"
    
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Choose optimal block sizes based on data type
    if A.dtype in [torch.float16, torch.bfloat16]:
        BLOCK_M = 128  # Larger blocks for better occupancy with FP16
        BLOCK_K = 256  # Larger K dimension for better memory coalescing
        kernel = matvec_kernel_fp16
    else:
        BLOCK_M = 64   # Balance register pressure for FP32
        BLOCK_K = 128  # Optimize for L1/L2 cache
        kernel = matvec_kernel
    
    # Ensure block sizes are powers of 2
    BLOCK_M = triton.next_power_of_2(min(BLOCK_M, M))
    BLOCK_K = triton.next_power_of_2(min(BLOCK_K, K))
    
    # Configure grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    
    # Launch kernel
    kernel[grid](
        A,
        B,
        C,
        M,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication using optimized Triton kernels.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return triton_matvec(A, B)
