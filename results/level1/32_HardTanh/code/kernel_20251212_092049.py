import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardtanh_kernel_1d(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized 1D HardTanh kernel with vectorized operations."""
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused min/max for hardtanh [-1, 1]
    out = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Vectorized store
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def hardtanh_kernel_2d(
    x_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """2D tiled kernel with proper stride handling."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load tile with proper striding
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Compute hardtanh
    out = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Store result
    out_ptrs = out_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])

def triton_hardtanh_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton HardTanh with adaptive grid strategy."""
    output = torch.empty_like(x)
    
    # Handle contiguous memory layout
    if not x.is_contiguous():
        x = x.contiguous()
        output = output.contiguous()
    
    if x.dim() <= 1:
        # 1D case: use simple 1D grid
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        hardtanh_kernel_1d[grid](
            x, output, n_elements,
            BLOCK_SIZE=1024
        )
    else:
        # 2D+ case: use 2D tiled kernel
        # Flatten all but last dimension for better memory access
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, x.shape[-1])
            output = output.view(-1, x.shape[-1])
        
        M, N = x.shape
        
        # Tune block sizes for better performance
        BLOCK_SIZE_M = 128 if M >= 128 else triton.next_power_of_2(M)
        BLOCK_SIZE_N = 128 if N >= 128 else triton.next_power_of_2(N)
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        hardtanh_kernel_2d[grid](
            x, output,
            x.stride(0), x.stride(1),
            output.stride(0), output.stride(1),
            M, N,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        
        if x.dim() > 2:
            output = output.view(original_shape)
    
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
