import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # High occupancy, balanced configs for Ada Lovelace
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        # Increased parallelism for smaller matrices
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        # Maximize tensor core utilization
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block starting indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers to blocks of A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Prefetch first block with correct mask shapes
    k_offset = 0
    a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k_offset) < K)
    b_mask = ((offs_k[:, None] + k_offset) < K) & (offs_n[None, :] < N)
    
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Prefetch next block if not last iteration
        if k < tl.cdiv(K, BLOCK_K) - 1:
            next_k_offset = (k + 1) * BLOCK_K
            next_a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + next_k_offset) < K)
            next_b_mask = ((offs_k[:, None] + next_k_offset) < K) & (offs_n[None, :] < N)
            next_a_ptrs = a_ptrs + BLOCK_K * stride_ak
            next_b_ptrs = b_ptrs + BLOCK_K * stride_bk
            next_a = tl.load(next_a_ptrs, mask=next_a_mask, other=0.0)
            next_b = tl.load(next_b_ptrs, mask=next_b_mask, other=0.0)
        
        # Compute with tensor cores
        if USE_TF32:
            accumulator += tl.dot(a, b, allow_tf32=True)
        else:
            accumulator += tl.dot(a, b)
        
        # Update pointers and registers for next iteration
        if k < tl.cdiv(K, BLOCK_K) - 1:
            a = next_a
            b = next_b
            a_ptrs = next_a_ptrs
            b_ptrs = next_b_ptrs
        else:
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    
    # Store result with coalesced writes
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(1), f"K dimension mismatch: {a.size(1)} != {b.size(1)}"
    
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b
    
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    
    if a.is_cuda:
        USE_TF32 = a.dtype == torch.float32
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
        
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            USE_TF32=USE_TF32,
        )
    else:
        c = torch.matmul(a, b.T)
    
    return c

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
