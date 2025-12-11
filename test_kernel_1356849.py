import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ak, stride_am,  # A is (K, M), strides: K-dim, M-dim
    stride_bk, stride_bn,  # B is (K, N)
    stride_cm, stride_cn,  # C is (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Optimized matrix multiplication kernel for A^T @ B with software pipelining"""
    # 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Precompute pointers for first iteration
    k = 0
    mask_k = (k + offs_k[:, None]) < K
    mask_a = mask_k & (offs_m[None, :] < M)
    mask_b = mask_k & (offs_n[None, :] < N)
    
    a_ptrs = a_ptr + ((k + offs_k[:, None]) * stride_ak + offs_m[None, :] * stride_am)
    b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
    
    # Prefetch first block
    a_block = tl.load(a_ptrs, mask=mask_a, other=0.0)
    b_block = tl.load(b_ptrs, mask=mask_b, other=0.0)
    
    # Main computation loop with software pipelining
    for k in range(BLOCK_K, K, BLOCK_K):
        # Prefetch next block
        next_k = k
        next_mask_k = (next_k + offs_k[:, None]) < K
        next_mask_a = next_mask_k & (offs_m[None, :] < M)
        next_mask_b = next_mask_k & (offs_n[None, :] < N)
        
        next_a_ptrs = a_ptr + ((next_k + offs_k[:, None]) * stride_ak + offs_m[None, :] * stride_am)
        next_b_ptrs = b_ptr + ((next_k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        
        next_a = tl.load(next_a_ptrs, mask=next_mask_a, other=0.0)
        next_b = tl.load(next_b_ptrs, mask=next_mask_b, other=0.0)
        
        # Compute with current block
        a_t = tl.trans(a_block)  # Now shape (BLOCK_M, BLOCK_K)
        acc += tl.dot(a_t, b_block, allow_tf32=True)
        
        # Swap buffers for next iteration
        a_block = next_a
        b_block = next_b
    
    # Compute final block
    if K > 0:
        a_t = tl.trans(a_block)
        acc += tl.dot(a_t, b_block, allow_tf32=True)
    
    # Store result
    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    mask_c = (offs_m < M) & (offs_n < N)
    c_ptrs = c_ptr + (offs_m * stride_cm + offs_n * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_c)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Triton matrix multiplication wrapper for A^T @ B"""
    # Check shapes
    assert A.shape[0] == B.shape[0], f"K dimension mismatch: {A.shape[0]} != {B.shape[0]}"
    
    K, M = A.shape
    _, N = B.shape
    
    # Output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Fixed block sizes optimized for Ada Lovelace
    # 128x128x16 balances register pressure and L2 locality
    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = 16  # Fixed for better L2 hit rate and reduced register pressure
    
    # Grid computation (2D)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Strides (A is K×M, B is K×N, C is M×N)
    stride_ak, stride_am = A.stride(0), A.stride(1)  # K-dim stride, M-dim stride
    stride_bk, stride_bn = B.stride(0), B.stride(1)  # K-dim stride, N-dim stride
    stride_cm, stride_cn = C.stride(0), C.stride(1)  # M-dim stride, N-dim stride
    
    # Launch kernel with optimized parameters
    # num_warps=4 reduces register pressure, num_stages=4 for better latency hiding
    matmul_kernel[grid](
        a_ptr=A,
        b_ptr=B,
        c_ptr=C,
        M=M,
        N=N,
        K=K,
        stride_ak=stride_ak,
        stride_am=stride_am,
        stride_bk=stride_bk,
        stride_bn=stride_bn,
        stride_cm=stride_cm,
        stride_cn=stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,      # Reduced for better occupancy
        num_stages=4,     # Increased for better latency hiding
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model using Triton for matrix multiplication
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)
