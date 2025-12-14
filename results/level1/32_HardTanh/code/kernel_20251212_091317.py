import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardtanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # HardTanh: clamp to [-1, 1]
    # Use min/max for direct computation - more efficient than conditional branching
    # First compute lower bound: max(x, -1)
    lower = tl.maximum(x, -1.0)
    # Then compute upper bound: min(lower, 1)
    out = tl.minimum(lower, 1.0)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardtanh(x: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated HardTanh activation function."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Use optimal block size for A100
    # 1024 threads maximizes occupancy while staying within register limits
    BLOCK_SIZE = 1024
    
    # Calculate grid size - 1D grid covering all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    hardtanh_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    """Optimized model using Triton kernels for HardTanh activation."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardTanh activation using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with HardTanh applied.
        """
        return triton_hardtanh(x)
