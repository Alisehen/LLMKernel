import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Balanced configurations for various matrix shapes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for A and B
    a_ptr += offs_m[:, None] * stride_am
    b_ptr += offs_n[None, :] * stride_bn
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Compute current K offset
        k_offset = k * BLOCK_K
        
        # Create masks for A and B
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k_offset) < K)
        b_mask = ((offs_k[:, None] + k_offset) < K) & (offs_n[None, :] < N)
        
        # Load A block with proper broadcasting for mask
        a = tl.load(
            a_ptr + (offs_k[None, :] + k_offset) * stride_ak,
            mask=a_mask,
            other=0.0
        )
        
        # Load B block with proper broadcasting for mask
        b = tl.load(
            b_ptr + (offs_k[:, None] + k_offset) * stride_bk,
            mask=b_mask,
            other=0.0
        )
        
        # Matrix multiplication with tensor cores
        if USE_TF32:
            accumulator += tl.dot(a, b, allow_tf32=True)
        else:
            accumulator += tl.dot(a, b)
    
    # Create mask for C output
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
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
            b.stride(1), b.stride(0),  # B is transposed: K becomes stride(1), N becomes stride(0)
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
