import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Compute Softplus activation: log(1 + exp(x))
    
    Optimized with:
    - Vectorized loads/stores for better memory throughput
    - Reduced branching overhead
    - Better SM utilization
    """
    pid = tl.program_id(axis=0)
    
    # Vectorized offsets for better memory coalescing
    base_offset = pid * BLOCK_SIZE * VEC_SIZE
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    
    # Create mask for all vector elements
    mask = offsets < n_elements
    
    # Vectorized load
    x_vec = tl.load(x_ptr + offsets, mask=mask)
    
    # Constants for numerical stability
    THRESHOLD_HIGH = 20.0
    THRESHOLD_LOW = -15.0
    
    # Vectorized computation
    # For large positive x: softplus(x) ≈ x
    # For large negative x: softplus(x) ≈ exp(x)
    
    # Create masks for different regimes
    high_mask = x_vec > THRESHOLD_HIGH
    low_mask = x_vec < THRESHOLD_LOW
    mid_mask = ~(high_mask | low_mask)
    
    # Compute for all regimes using vectorized operations
    exp_x = tl.where(mid_mask, tl.exp(x_vec), 0.0)
    softplus_mid = tl.where(mid_mask, tl.log(1.0 + exp_x), 0.0)
    
    # High range: x
    softplus_high = tl.where(high_mask, x_vec, 0.0)
    
    # Low range: exp(x)
    softplus_low = tl.where(low_mask, tl.exp(x_vec), 0.0)
    
    # Combine all regimes
    output = softplus_mid + softplus_high + softplus_low
    
    # Vectorized store
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    """
    Softplus activation wrapper for Triton kernel.
    
    Args:
        x: Input tensor of any shape
        
    Returns:
        Tensor with Softplus applied, same shape as input
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized grid calculation for better SM utilization
    # Use vectorization to reduce grid size and increase work per thread
    VEC_SIZE = 4  # Process 4 elements per thread for better memory throughput
    
    # Dynamically calculate optimal BLOCK_SIZE based on tensor size
    # Target 4096-8192 total threads per SM for optimal occupancy
    if n_elements >= 4194304:  # 4M+ elements
        BLOCK_SIZE = 256  # 8 warps per block
    elif n_elements >= 1048576:  # 1M+ elements
        BLOCK_SIZE = 128  # 4 warps per block
    else:
        BLOCK_SIZE = 64   # 2 warps per block
    
    # Calculate grid size with vectorization
    total_work_items = triton.cdiv(n_elements, VEC_SIZE)
    grid = (triton.cdiv(total_work_items, BLOCK_SIZE),)
    
    # Adjust num_warps based on BLOCK_SIZE for optimal occupancy
    num_warps = BLOCK_SIZE // 32
    
    # Launch kernel with optimized configuration
    softplus_kernel[grid](
        x, 
        output, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        num_warps=num_warps,
        num_stages=3
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using Triton.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        return triton_softplus(x)
