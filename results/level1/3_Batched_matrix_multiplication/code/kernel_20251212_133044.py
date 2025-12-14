import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=1),
    ],
    key=['M', 'N', 'K', 'batch_size'],
)
@triton.jit
def bmm_kernel_optimized(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    batch_size,
    stride_Ab, stride_Am, stride_Ak,
    stride_Bb, stride_Bk, stride_Bn,
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized batched matrix multiplication kernel.
    Removed prefetching (tl.prefetch doesn't exist) and focused on efficient tiling.
    """
    pid_batch = tl.program_id(2)
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptr = A_ptr + pid_batch * stride_Ab
    B_ptr = B_ptr + pid_batch * stride_Bb
    C_ptr = C_ptr + pid_batch * stride_Cb
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_effective = min(BLOCK_K, k_remaining)
        
        # Load current A tile
        a_ptrs = A_ptr + (offs_m[:, None] * stride_Am + (k * BLOCK_K + offs_k[None, :]) * stride_Ak)
        a_mask = (offs_m[:, None] < M) & ((k * BLOCK_K + offs_k[None, :]) < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load current B tile
        b_ptrs = B_ptr + ((k * BLOCK_K + offs_k[:, None]) * stride_Bk + offs_n[None, :] * stride_Bn)
        b_mask = ((k * BLOCK_K + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate using matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
    
    # Write back results
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_Cm + offs_cn[None, :] * stride_Cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_bmm_optimized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3D tensors"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, M, K = A.shape
    _, _, N = B.shape
    
    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        1,
        batch_size,
    )
    
    bmm_kernel_optimized[grid](
        A, B, C,
        M, N, K,
        batch_size,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    
    return C

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm_optimized(A, B)
