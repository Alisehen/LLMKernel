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
    EVEN_K: tl.constexpr,
):
    """
    Optimized upper triangular matrix multiplication kernel.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Boundary masks
    m_mask = offs_m < N
    n_mask = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute valid K range for this block
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N
    k_start = tl.maximum(start_m, start_n)
    
    # Main computation loop
    for k in range(k_start, N, BLOCK_K):
        k_remaining = N - k
        k_valid = tl.minimum(BLOCK_K, k_remaining)
        
        # Load A block
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & ((k + offs_k[None, :]) < N), other=0.0)
        
        # Load B block  
        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < N) & n_mask[None, :], other=0.0)
        
        # Apply triangular conditions
        if EVEN_K:
            # Efficient triangular masking with broadcasting
            m_idx = offs_m[:, None]  # (BLOCK_M, 1)
            n_idx = offs_n[None, :]  # (1, BLOCK_N)
            k_idx = k + offs_k       # (BLOCK_K,)
            
            # Upper triangular conditions
            a_mask = (m_idx <= k_idx[None, :])  # (BLOCK_M, BLOCK_K)
            b_mask = (k_idx[:, None] <= n_idx)  # (BLOCK_K, BLOCK_N)
            
            a = tl.where(a_mask, a, 0.0)
            b = tl.where(b_mask, b, 0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Final upper triangular condition and store
    row_cond = offs_m[:, None] <= offs_n[None, :]
    acc = tl.where(row_cond, acc, 0.0)
    
    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized Triton implementation of upper triangular matrix multiplication.
    """
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape == B.shape
    N = A.shape[0]
    
    # Allocate output
    C = torch.zeros((N, N), device=A.device, dtype=A.dtype)
    
    # Autotuning configurations
    configs = []
    for num_stages in [3, 4]:
        for BLOCK_M in [64, 128]:
            for BLOCK_N in [64, 128]:
                for BLOCK_K in [32, 64, 128]:
                    for num_warps in [4, 8]:
                        if BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N <= 49152:
                            configs.append(triton.Config({
                                'BLOCK_M': BLOCK_M,
                                'BLOCK_N': BLOCK_N,
                                'BLOCK_K': BLOCK_K,
                                'EVEN_K': (N % BLOCK_K == 0),
                            }, num_stages=num_stages, num_warps=num_warps))
    
    @triton.autotune(configs=configs, key=['N'])
    @triton.jit
    def kernel_wrapper(
        A_ptr, B_ptr, C_ptr, N,
        stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
    ):
        triu_matmul_kernel(
            A_ptr, B_ptr, C_ptr, N,
            stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K,
        )
    
    # Launch kernel
    grid = (triton.cdiv(N, 64), triton.cdiv(N, 64))
    kernel_wrapper[grid](
        A, B, C, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model for upper triangular matrix multiplication using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication for upper triangular matrices.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
            
        Returns:
            torch.Tensor: The product of A and B, also upper triangular.
        """
        return triton_triu_matmul(A, B)
