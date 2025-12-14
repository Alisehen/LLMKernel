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
    
    # Calculate block boundaries
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    
    # Create offsets for the block
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    
    # Create masks for boundary conditions
    mask_m = offs_m < N
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, N, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        mask_k = k_offs < N
        
        # Load A block - upper triangular condition
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :] & (offs_m[:, None] <= k_offs[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B block - upper triangular condition  
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :] & (k_offs[:, None] <= offs_n[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b, allow_tf32=False)
    
    # Apply upper triangular condition: only store if row <= column
    row_cond = offs_m[:, None] <= offs_n[None, :]
    acc = tl.where(row_cond, acc, 0.0)
    
    # Store result with proper masking
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of upper triangular matrix multiplication.
    """
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape == B.shape
    N = A.shape[0]
    
    # Allocate output tensor initialized to zeros
    C = torch.zeros((N, N), device=A.device, dtype=A.dtype)
    
    # Autotune configurations for different block sizes
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ]
    
    # Find best configuration using autotune
    best_config = None
    best_time = float('inf')
    
    for config in configs:
        BLOCK_M = config.kwargs['BLOCK_M']
        BLOCK_N = config.kwargs['BLOCK_N']
        BLOCK_K = config.kwargs['BLOCK_K']
        
        # Adjust block sizes if N is smaller than block size
        block_m = min(BLOCK_M, triton.next_power_of_2(N))
        block_n = min(BLOCK_N, triton.next_power_of_2(N))
        block_k = min(BLOCK_K, triton.next_power_of_2(N))
        
        grid = (triton.cdiv(N, block_m), triton.cdiv(N, block_n))
        
        # Warmup run
        triu_matmul_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )
        
        # Benchmark run
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        triu_matmul_kernel[grid](
            A, B, C,
            N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )
        end.record()
        torch.cuda.synchronize()
        
        elapsed = start.elapsed_time(end)
        if elapsed < best_time:
            best_time = elapsed
            best_config = config
    
    # Use best configuration for final computation
    BLOCK_M = best_config.kwargs['BLOCK_M']
    BLOCK_N = best_config.kwargs['BLOCK_N']
    BLOCK_K = best_config.kwargs['BLOCK_K']
    
    # Adjust block sizes if N is smaller than block size
    block_m = min(BLOCK_M, triton.next_power_of_2(N))
    block_n = min(BLOCK_N, triton.next_power_of_2(N))
    block_k = min(BLOCK_K, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, block_m), triton.cdiv(N, block_n))
    
    # Clear output tensor for final computation
    C.zero_()
    
    triu_matmul_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_stages=best_config.num_stages,
        num_warps=best_config.num_warps,
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
