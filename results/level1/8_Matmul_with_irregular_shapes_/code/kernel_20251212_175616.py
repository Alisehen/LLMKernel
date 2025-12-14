import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Group size for dot product optimization
    GROUP_M: tl.constexpr,
    # Whether to use Tensor Cores
    USE_TENSOR_CORES: tl.constexpr,
):
    """Optimized matrix multiplication kernel with support for irregular shapes."""
    
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
    
    # Offsets for tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Main K-loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles with masking
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        
        # Load A tile
        a_mask = (offs_m[:, None] < M) & k_mask
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply
        if USE_TENSOR_CORES:
            # Use tensor cores for higher throughput
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        else:
            # Standard matmul for non-tensor core cases
            acc += tl.dot(a, b)
        
        # Update pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton-optimized matrix multiplication."""
    # Check and ensure 2D tensors
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Get dimensions
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # Determine optimal tile sizes based on hardware and problem size
    # Using parameters optimized for Ampere architecture (compute capability 8.9)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    GROUP_M = 8
    
    # Check if we can use tensor cores
    USE_TENSOR_CORES = a.dtype in [torch.float16, torch.bfloat16] and BLOCK_K % 16 == 0
    
    # Calculate grid size
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        USE_TENSOR_CORES=USE_TENSOR_CORES,
    )
    
    return c

class ModelNew(nn.Module):
    """Optimized model using Triton for matrix multiplication."""
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs matrix multiplication of A and B using Triton kernels."""
        return triton_matmul(A, B)
