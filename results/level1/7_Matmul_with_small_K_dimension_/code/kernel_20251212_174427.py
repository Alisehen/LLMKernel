import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Simplified kernel for small K dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to A and B blocks
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # For small K, we can load everything in one iteration
    if tl.cdiv(K, BLOCK_K) == 1:
        # Load A and B once
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)
    else:
        # Loop over K dimension
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((offs_k[None, :] + k) < K), other=0.0)
            b = tl.load(b_ptrs, mask=((offs_k[:, None] + k) < K) & (offs_n[None, :] < N), other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
    # Write back result
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K, N = b.shape
    
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.dtype == b.dtype == torch.float32, "Only float32 supported"
    
    # Ensure contiguous memory layout
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Choose block sizes based on problem dimensions
    if K <= 64:
        # For small K, use larger blocks for M and N
        BLOCK_M = 128 if M >= 128 else 64
        BLOCK_N = 128 if N >= 128 else 64
        BLOCK_K = K
    else:
        # For larger K, use smaller blocks to fit in shared memory
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
    
    # Adjust block sizes for better GPU utilization
    if BLOCK_M * BLOCK_K > 8192:  # Limit shared memory usage
        BLOCK_M = 64
    if BLOCK_N * BLOCK_K > 8192:
        BLOCK_N = 64
    
    # Calculate grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel with optimized parameters
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4 if K <= 32 else 8,
        num_stages=2,  # Reduced to avoid shared memory overflow
    )
    
    return c


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)
