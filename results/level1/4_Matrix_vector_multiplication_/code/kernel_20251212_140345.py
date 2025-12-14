import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256, 'VEC_A': 4, 'VEC_B': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'VEC_A': 4, 'VEC_B': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256, 'VEC_A': 4, 'VEC_B': 4}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'VEC_A': 2, 'VEC_B': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64, 'VEC_A': 2, 'VEC_B': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256, 'VEC_A': 1, 'VEC_B': 1}, num_warps=4, num_stages=2),
    ],
    key=['M', 'K'],
)
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
    VEC_A: tl.constexpr,
    VEC_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    
    # Pre-allocate accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Pre-compute block offsets for K
    K_blocks = tl.cdiv(K, BLOCK_K)
    
    # Loop with software pipelining
    for k in range(0, K_blocks):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Vectorized loads with prefetch
        if VEC_A == 4:
            # Load A in vectors of 4
            A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :], 
                       cache_modifier=tl.CacheModifier.PREFETCH)
        else:
            # Scalar load
            A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
        
        if VEC_B == 4:
            # Load B in vectors of 4
            B_ptrs = B_ptr + offs_k * stride_bk
            b = tl.load(B_ptrs, mask=k_mask, cache_modifier=tl.CacheModifier.PREFETCH)
        else:
            # Scalar load
            B_ptrs = B_ptr + offs_k * stride_bk
            b = tl.load(B_ptrs, mask=k_mask)
        
        # Fused multiply-accumulate
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store with vectorization when possible
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.store(C_ptrs, acc, mask=m_mask)


@triton.jit
def matvec_kernel_shared(
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
    pid = tl.program_id(0)
    
    # Allocate shared memory for B reuse
    B_shm = tl.alloc(tl.float32, BLOCK_K)
    
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    K_blocks = tl.cdiv(K, BLOCK_K)
    
    for k in range(0, K_blocks):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Load B into shared memory once per block
        B_ptrs = B_ptr + offs_k * stride_bk
        b = tl.load(B_ptrs, mask=k_mask, cache_modifier=tl.CacheModifier.PREFETCH)
        tl.store(B_shm + tl.arange(0, BLOCK_K), b)
        tl.debug_barrier()
        
        # Load A with coalesced access pattern
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
        
        # Read B from shared memory
        b_shm = tl.load(B_shm + tl.arange(0, BLOCK_K))
        
        # Compute
        acc += tl.sum(a * b_shm[None, :], axis=1)
        tl.debug_barrier()
    
    # Store result
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.store(C_ptrs, acc, mask=m_mask)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert B.shape == (K, 1), f"B must be shape ({K}, 1), got {B.shape}"
    
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Use appropriate kernel based on problem size
    if K >= 8192:  # Large K benefits from shared memory
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
        matvec_kernel_shared[grid](
            A, B.view(-1), C.view(-1),
            M, K,
            A.stride(0), A.stride(1),
            B.stride(0), C.stride(0),
            BLOCK_M=128, BLOCK_K=256
        )
    else:
        # Auto-tuned kernel for smaller problems
        A_acc = A if A.dtype == torch.float32 else A.to(torch.float32)
        B_acc = B if B.dtype == torch.float32 else B.to(torch.float32)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
        
        matvec_kernel[grid](
            A_acc,
            B_acc.view(-1),
            C if C.dtype == torch.float32 else C.to(torch.float32).view(-1),
            M, K,
            A_acc.stride(0), A_acc.stride(1),
            B_acc.stride(0),
            C.stride(0),
        )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)
