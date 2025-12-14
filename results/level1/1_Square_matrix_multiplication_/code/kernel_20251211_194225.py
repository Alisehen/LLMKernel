import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 2D grid layout for better SM distribution
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    # Adjust for tile grouping in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    group_id = pid_m // GROUP_SIZE_M
    group_size = min(GROUP_SIZE_M, num_pid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid_m % group_size)
    
    # Skip if out of bounds
    if pid_m >= num_pid_m:
        return
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        
        acc += tl.dot(a, b)
        
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    
    # Store result
    C = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


@triton.jit
def matmul_kernel_tf32(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 2D grid layout for better SM distribution
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    # Adjust for tile grouping in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    group_id = pid_m // GROUP_SIZE_M
    group_size = min(GROUP_SIZE_M, num_pid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid_m % group_size)
    
    # Skip if out of bounds
    if pid_m >= num_pid_m:
        return
    
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < K - k), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < K - k) & (rn[None, :] < N), other=0.0)
        
        acc += tl.dot(a, b)
        
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    
    # Store result
    C = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    if a.dtype == torch.float32 and b.dtype == torch.float32:
        # Optimized grid calculation for 2D distribution
        def grid(META):
            total_blocks = (triton.cdiv(M, META['BLOCK_SIZE_M']) * 
                          triton.cdiv(N, META['BLOCK_SIZE_N']))
            return (total_blocks,)
        
        # Autotune configurations for better SM utilization
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=8, num_stages=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=8, num_stages=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_warps=4, num_stages=3),
            ],
            key=['M', 'N', 'K'],
        )
        def matmul_config_selector(M, N, K, **kwargs):
            pass
        
        if torch.backends.cuda.matmul.allow_tf32:
            matmul_kernel_tf32[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,
                num_stages=4,
                num_warps=8,
            )
        else:
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=128,
                BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=32,
                GROUP_SIZE_M=8,
                num_stages=4,
                num_warps=8,
            )
    else:
        c = torch.matmul(a, b)
    
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
