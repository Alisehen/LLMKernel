import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_stages': 3}, num_warps=4),
        triton.Config({'num_stages': 4}, num_warps=4),
        triton.Config({'num_stages': 3}, num_warps=8),
        triton.Config({'num_stages': 4}, num_warps=8),
    ],
    key=['M', 'N']
)
@triton.jit
def mul_scalar_kernel_2d(
    a_ptr,
    s,
    out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Optimized 2D kernel with pipelining for non-contiguous tensors"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Precompute offsets for better ILP
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks with early exit
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Compute base pointers with strength reduction
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    
    # Load and compute with pipelining
    a_block = tl.load(a_ptrs, mask=mask, other=0.0)
    out_block = a_block * s
    tl.store(out_ptrs, out_block, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'num_stages': 3}, num_warps=8),
        triton.Config({'num_stages': 4}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=4),
        triton.Config({'num_stages': 4}, num_warps=4),
    ],
    key=['n_elements']
)
@triton.jit
def mul_scalar_kernel_1d_vectorized(
    a_ptr,
    s,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
    VECTOR_SIZE: tl.constexpr = 4,
):
    """Vectorized 1D kernel with optimized memory access patterns"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * VECTOR_SIZE
    
    # Vectorized offsets for better memory coalescing
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VECTOR_SIZE + tl.arange(0, VECTOR_SIZE)[None, :]
    mask = offsets < n_elements
    
    # Vectorized load/store for better bandwidth utilization
    a_vec = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    out_vec = a_vec * s
    tl.store(out_ptr + offsets, out_vec, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'num_stages': 2}, num_warps=32),
        triton.Config({'num_stages': 3}, num_warps=32),
        triton.Config({'num_stages': 2}, num_warps=16),
        triton.Config({'num_stages': 3}, num_warps=16),
    ],
    key=['n_elements']
)
@triton.jit
def mul_scalar_kernel_1d_optimized(
    a_ptr,
    s,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Optimized 1D kernel with improved occupancy and latency hiding"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Use larger block with multiple prefetch stages
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Prefetch hint for better cache utilization
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca")
    output = a * s
    tl.store(out_ptr + offsets, output, mask=mask, cache_modifier=".cg")


def triton_mul_scalar(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Optimized matrix-scalar multiplication with adaptive kernel selection.
    
    Args:
        A: Input matrix of shape (M, N)
        s: Scalar value
        
    Returns:
        Resulting matrix of shape (M, N)
    """
    output = torch.empty_like(A)
    
    if A.is_contiguous():
        n_elements = A.numel()
        
        # Choose kernel based on problem size
        if n_elements >= 1048576:  # Large problem - use vectorized kernel
            BLOCK_SIZE = 256
            VECTOR_SIZE = 4
            grid = (triton.cdiv(n_elements, BLOCK_SIZE * VECTOR_SIZE),)
            
            mul_scalar_kernel_1d_vectorized[grid](
                A, s, output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                VECTOR_SIZE=VECTOR_SIZE
            )
        else:  # Small to medium problem - use optimized 1D kernel
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            mul_scalar_kernel_1d_optimized[grid](
                A, s, output, n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        # Use 2D kernel for non-contiguous tensors
        M, N = A.shape
        stride_am, stride_an = A.stride()
        stride_om, stride_on = output.stride()
        
        # Tuned block sizes for Ada Lovelace
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 32
        
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N)
        )
        
        mul_scalar_kernel_2d[grid](
            A, s, output, M, N,
            stride_am, stride_an, stride_om, stride_on,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix-scalar multiplication using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using optimized Triton kernels.
        
        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value
            
        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return triton_mul_scalar(A, s)
