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
    # Compute program IDs
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Group programs to improve L2 cache hit rate
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Compute program IDs within group
    pid_m = first_pid_m + ((pid % num_pid_in_group) // num_pid_n)
    pid_n = (pid % num_pid_in_group) % num_pid_n
    
    # Offsets for blocks
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Create block pointers with proper strides
    A = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load blocks with masking
        a = tl.load(A, mask=(rm[:, None] < M) & ((k * BLOCK_SIZE_K + rk[None, :]) < K), other=0.0)
        b = tl.load(B, mask=((k * BLOCK_SIZE_K + rk[:, None]) < K) & (rn[None, :] < N), other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Move to next blocks
        A += BLOCK_SIZE_K * stride_ak
        B += BLOCK_SIZE_K * stride_bk
    
    # Store result
    C = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Autotuner configurations
configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
]


@triton.autotune(
    configs=configs,
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(
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
    # Call the main kernel logic
    matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        BLOCK_SIZE_K, GROUP_SIZE_M
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    if a.dtype == torch.float32 and b.dtype == torch.float32:
        # Grid calculation function
        def grid(meta):
            num_pid_m = triton.cdiv(M, meta['BLOCK_SIZE_M'])
            num_pid_n = triton.cdiv(N, meta['BLOCK_SIZE_N'])
            num_pid_in_group = meta['GROUP_SIZE_M'] * num_pid_n
            num_groups = triton.cdiv(num_pid_m, meta['GROUP_SIZE_M'])
            return (num_groups * num_pid_in_group,)
        
        # Launch kernel
        matmul_kernel_autotuned[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
    else:
        # Fallback for non-float32
        c = torch.matmul(a, b)
    
    return c


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
