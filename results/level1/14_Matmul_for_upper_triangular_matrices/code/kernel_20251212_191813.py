import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triu_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
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
    """
    Matrix multiplication kernel that only computes upper triangular part C = triu(A @ B).
    A and B are upper triangular matrices.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create pointers for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for boundary conditions
    mask_m = offs_m < N
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Upper triangular condition: only compute if row <= column
    row_cond = offs_m[:, None] <= offs_n[None, :]
    
    # Skip lower triangular blocks entirely
    if not tl.any(row_cond):
        return
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N
        
        # Load A block
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load B block  
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Update accumulators
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Apply upper triangular mask and store
    acc = tl.where(row_cond, acc, 0.0)
    
    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of upper triangular matrix multiplication.
    """
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape == B.shape
    N = A.shape[0]
    
    # Allocate output tensor
    C = torch.empty((N, N), device=A.device, dtype=A.dtype)
    
    # Set block sizes that fit within shared memory constraints
    # Using smaller blocks to avoid shared memory overflow
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_stages=3,
        num_warps=4,
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication for upper triangular matrices using Triton.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
            
        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        return triton_triu_matmul(A, B)
