import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_tall_skinny(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # 2D block grid - optimized for M >> N case
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Offsets for the output tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute dot products for this tile
    # Since N is small (32), we can process all columns in one go
    for k in range(0, N):
        # Load A tile (BLOCK_SIZE_M x 1) - one column at a time
        a_offs = offs_m * stride_am + k * stride_an
        a_mask = offs_m < M
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        
        # Load B tile (1 x BLOCK_SIZE_N) - one row at a time  
        b_offs = k * stride_bn + offs_n * stride_bm
        b_mask = offs_n < M
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
        
        # Outer product accumulation
        acc += a[:, None] * b[None, :]
    
    # Store result tile
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < M)
    tl.store(c_ptr + c_offs, acc, mask=c_mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M = A.size(0)  # 32768
    N = A.size(1)  # 32
    assert B.size(0) == N and B.size(1) == M, "Matrix dimensions don't match"
    
    # Preallocate output
    C = torch.empty((M, M), device=A.device, dtype=A.dtype)
    
    # Choose block sizes - optimized for tall-skinny case
    BLOCK_SIZE_M = 128  # Larger block for better memory efficiency on M dimension
    BLOCK_SIZE_N = 64   # Smaller block for N dimension since N is small
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(M, meta['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_kernel_tall_skinny[grid](
        A,
        B,
        C,
        M,
        N,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for tall-skinny matrix multiplication.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using optimized Triton kernel.
        
        Args:
            A (torch.Tensor): Input matrix of shape (M, N) where M >> N
            B (torch.Tensor): Input matrix of shape (N, M) where M >> N
        
        Returns:
            torch.Tensor: Output matrix of shape (M, M)
        """
        return triton_matmul(A, B)
