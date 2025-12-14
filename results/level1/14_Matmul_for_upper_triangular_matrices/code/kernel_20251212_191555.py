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
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for boundary conditions
    mask_m = offs_m < N
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Upper triangular condition: only compute if row <= column
    row_cond = offs_m[:, None] <= offs_n[None, :]
    
    # Load blocks of A and B
    A_block_ptr = tl.make_block_ptr(
        A_ptr,
        shape=(N, N),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    B_block_ptr = tl.make_block_ptr(
        B_ptr,
        shape=(N, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_K):
        # Load A block - upper triangular optimization
        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Load B block - upper triangular optimization  
        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        # Update accumulators
        acc += tl.dot(a, b)
        
        # Update block pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
    
    # Apply upper triangular mask and store
    acc = tl.where(row_cond, acc, 0.0)
    
    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def triu_matmul_fused_kernel(
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
    Fused kernel that computes matmul with upper triangular optimization.
    Uses tensor cores for better performance on Ampere GPUs.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Upper triangular condition: skip computation if row > column
    if pid_m > pid_n:
        return
    
    # Create block pointers with upper triangular optimization
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    B_ptrs = B_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Masks
    mask_a = (offs_m[:, None] < N) & (tl.arange(0, BLOCK_K)[None, :] < N)
    mask_b = (tl.arange(0, BLOCK_K)[:, None] < N) & (offs_n[None, :] < N)
    
    # Initialize accumulator using tensor core precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop with optimized tile size for tensor cores
    for k in range(0, N, BLOCK_K):
        # Load tiles with boundary checks
        a = tl.load(A_ptrs, mask=mask_a, other=0.0)
        b = tl.load(B_ptrs, mask=mask_b, other=0.0)
        
        # Accumulate using tensor cores (TF32 on Ampere)
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Update pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        
        # Update masks
        mask_a = (offs_m[:, None] < N) & (tl.arange(0, BLOCK_K)[None, :] + k + BLOCK_K < N)
        mask_b = (tl.arange(0, BLOCK_K)[:, None] + k + BLOCK_K < N) & (offs_n[None, :] < N)
    
    # Only store if within upper triangular region
    row_mask = offs_m[:, None] <= offs_n[None, :]
    col_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    store_mask = row_mask & col_mask
    
    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=store_mask)

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of upper triangular matrix multiplication.
    """
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape == B.shape
    N = A.shape[0]
    
    # Allocate output tensor
    C = torch.empty((N, N), device=A.device, dtype=A.dtype)
    
    # Choose kernel based on matrix size
    if N >= 2048:
        # Use fused kernel with larger blocks for better tensor core utilization
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        triu_matmul_fused_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    else:
        # Use regular kernel with smaller blocks for better occupancy
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
