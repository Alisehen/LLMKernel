import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
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
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    
    # Pre-allocate accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Loop over K dimension with blocking
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Load B vector block
        B_ptrs = B_ptr + offs_k * stride_bk
        b = tl.load(B_ptrs, mask=k_mask)
        
        # Load A matrix block
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :])
        
        # Compute partial dot product
        acc += tl.sum(a * b[None, :], axis=1)
    
    # Store result
    C_ptrs = C_ptr + offs_m * stride_cm
    tl.store(C_ptrs, acc, mask=m_mask)


def triton_matvec(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    assert B.shape == (K, 1), f"B must be shape ({K}, 1), got {B.shape}"
    
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    # Convert to float32 for accumulation if needed
    if A.dtype != torch.float32:
        A_acc = A.to(torch.float32)
        B_acc = B.to(torch.float32)
        C_acc = torch.empty((M, 1), device=A.device, dtype=torch.float32)
    else:
        A_acc = A
        B_acc = B
        C_acc = C
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    
    matvec_kernel[grid](
        A_acc,
        B_acc.view(-1),
        C_acc.view(-1),
        M, K,
        A_acc.stride(0), A_acc.stride(1),
        B_acc.stride(0),
        C_acc.stride(0),
    )
    
    # Convert back to original dtype if needed
    if A.dtype != torch.float32:
        C = C_acc.to(A.dtype)
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matvec(A, B)
