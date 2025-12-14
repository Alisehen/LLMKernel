import torch
import torch.nn as nn
import triton
import triton.language as tl

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
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load block of A with boundary check
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load block of B with boundary check
        b_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply and accumulate
        accumulator += tl.dot(a, b, allow_tf32=USE_TF32)
        
        # Move pointers to next block in K dimension
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write back result with boundary check
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Shape mismatch: a={a.shape}, b={b.shape}"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32 if a.dtype == torch.float32 else 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        USE_TF32=True,
        num_warps=8,
        num_stages=3,
    )
    
    return c

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        return triton_matmul(A, B)
