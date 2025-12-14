import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def hardtanh_kernel_optimized(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized HardTanh kernel with 1D grid for maximum parallelism."""
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load for better memory throughput
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused min/max for hardtanh [-1, 1]
    out = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Vectorized store
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def hardtanh_kernel_2d(
    x_ptr,
    out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """2D tiled kernel for better memory coalescing on 2D+ tensors."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block pointers for better memory access patterns
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for boundary checks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load tile with broadcasting for efficient memory access
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Compute hardtanh
    out = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Store result
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])

def triton_hardtanh_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton HardTanh with adaptive grid strategy."""
    output = torch.empty_like(x)
    
    # Handle different tensor dimensions
    if x.dim() <= 1:
        # 1D case: use simple 1D grid
        n_elements = output.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        hardtanh_kernel_optimized[grid](
            x, output, n_elements,
            BLOCK_SIZE=1024  # Will be overridden by autotune
        )
    elif x.dim() == 2:
        # 2D case: use 2D tiled kernel for better coalescing
        M, N = x.shape
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        
        hardtanh_kernel_2d[grid](
            x, output,
            M, N,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=128  # Will be overridden by autotune
        )
    else:
        # Higher dimensions: flatten to 2D for better parallelism
        # Flatten all dimensions except the last one
        orig_shape = x.shape
        M = x.numel() // orig_shape[-1]
        N = orig_shape[-1]
        
        x_2d = x.view(M, N)
        output_2d = output.view(M, N)
        
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        
        hardtanh_kernel_2d[grid](
            x_2d, output_2d,
            M, N,
            x_2d.stride(0), x_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            BLOCK_M=128, BLOCK_N=128
        )
    
    return output

class ModelNew(nn.Module):
    """Optimized model using adaptive Triton kernels for HardTanh activation."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation using optimized Triton kernel.
        Uses adaptive grid strategy based on input tensor dimensions.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with HardTanh applied.
        """
        return triton_hardtanh_optimized(x)
