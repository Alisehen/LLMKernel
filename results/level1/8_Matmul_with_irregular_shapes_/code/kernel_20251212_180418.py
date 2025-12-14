import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=2),
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
    GROUP_M: tl.constexpr,
):
    """Optimized matrix multiplication kernel with improved tile sizes and resource usage."""
    
    # Program ID grid
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets for tiles - use X dimension for more efficient memory access
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pre-calculate pointer increments
    a_ptr_base = a_ptr + (offs_m[:, None] * stride_am)
    b_ptr_base = b_ptr + (offs_n[None, :] * stride_bn)
    
    # Main K-loop with software pipelining
    for k in range(0, K, BLOCK_K):
        # Calculate remaining K for masking
        k_remaining = K - k
        
        # Load A tile with efficient masking
        a_mask = offs_m[:, None] < M
        a = tl.load(
            a_ptr_base + (offs_k[None, :] * stride_ak), 
            mask=a_mask & (offs_k[None, :] < k_remaining), 
            other=0.0
        )
        
        # Load B tile with efficient masking
        b_mask = offs_n[None, :] < N
        b = tl.load(
            b_ptr_base + (offs_k[:, None] * stride_bk), 
            mask=b_mask & (offs_k[:, None] < k_remaining), 
            other=0.0
        )
        
        # Accumulate matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Update base pointers for next K block
        a_ptr_base += BLOCK_K * stride_ak
        b_ptr_base += BLOCK_K * stride_bk
    
    # Store result with efficient masking
    c_ptr_block = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr_block, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton-optimized matrix multiplication with autotuning."""
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # Define grid function for autotuning
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    # Launch kernel with autotuning
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        GROUP_M=8,
    )
    
    return c

class ModelNew(nn.Module):
    """Optimized model using Triton for matrix multiplication."""
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs matrix multiplication of A and B using Triton kernels."""
        return triton_matmul(A, B)
