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
):
    """
    Compute Softplus activation: log(1 + exp(x))
    
    Using numerical stability improvements:
    - For x > 20: Use x directly (exp(x) overflows in fp32)
    - For x < -15: Use exp(x) directly (exp(x) underflows)
    - Else: Use log(1 + exp(x))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with masking
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Constants for numerical stability
    THRESHOLD_HIGH = 20.0
    THRESHOLD_LOW = -15.0
    
    # Compute softplus with numerical stability
    # For large positive x: softplus(x) ≈ x
    # For large negative x: softplus(x) ≈ exp(x)
    
    # Create masks for different regimes
    high_mask = x > THRESHOLD_HIGH
    low_mask = x < THRESHOLD_LOW
    mid_mask = ~(high_mask | low_mask)
    
    # Compute for different regimes
    # Medium range: log(1 + exp(x))
    exp_x = tl.where(mid_mask, tl.exp(x), 0.0)
    softplus_mid = tl.where(mid_mask, tl.log(1.0 + exp_x), 0.0)
    
    # High range: x
    softplus_high = tl.where(high_mask, x, 0.0)
    
    # Low range: exp(x)
    softplus_low = tl.where(low_mask, tl.exp(x), 0.0)
    
    # Combine all regimes
    output = softplus_mid + softplus_high + softplus_low
    
    # Store result
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
    
    # Heuristic for optimal block size based on tensor size
    if n_elements >= 1048576:  # 1M+ elements
        BLOCK_SIZE = 1024
    elif n_elements >= 262144:  # 256K+ elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Use 1D grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    softplus_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
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
