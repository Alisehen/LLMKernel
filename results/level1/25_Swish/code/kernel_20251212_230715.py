import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Swish activation: x * sigmoid(x) with stable formulation."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with mask
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Stable sigmoid implementation: 1/(1 + exp(-x))
    # For positive x: 1/(1 + exp(-x))
    # For negative x: exp(x)/(1 + exp(x))
    # Use sign-based optimization for numerical stability
    abs_x = tl.abs(x)
    exp_neg_abs_x = tl.exp(-abs_x)  # exp(-|x|)
    
    # Compute denominator: 1 + exp(-|x|)
    denom = 1.0 + exp_neg_abs_x
    
    # Compute numerator based on sign
    # For x >= 0: numerator = 1, so sigmoid = 1/denom = 1/(1 + exp(-x))
    # For x < 0: numerator = exp(x) = exp(-|x|), so sigmoid = exp(-|x|)/(1 + exp(-|x|))
    numerator = tl.where(x >= 0, 1.0, exp_neg_abs_x)
    sigmoid_x = numerator / denom
    
    # Swish: x * sigmoid(x)
    output = x * sigmoid_x
    
    # Store with mask
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def swish_fast_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_FAST_MATH: tl.constexpr,
):
    """Swish activation with optional fast math approximation."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if USE_FAST_MATH:
        # Fast approximation: x * (0.5 + 0.5 * tanh(x/2))
        # tanh(x/2) = (exp(x) - 1)/(exp(x) + 1) with numerical stability
        half_x = x * 0.5
        # Compute exp(x) with clipping for stability
        exp_x = tl.exp(tl.where(half_x > 10.0, 10.0, tl.where(half_x < -10.0, -10.0, half_x)))
        tanh_half_x = (exp_x - 1.0) / (exp_x + 1.0)
        sigmoid_approx = 0.5 + 0.5 * tanh_half_x
        output = x * sigmoid_approx
    else:
        # Standard stable implementation
        abs_x = tl.abs(x)
        exp_neg_abs_x = tl.exp(-abs_x)
        denom = 1.0 + exp_neg_abs_x
        numerator = tl.where(x >= 0, 1.0, exp_neg_abs_x)
        sigmoid_x = numerator / denom
        output = x * sigmoid_x
    
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_swish(x: torch.Tensor, fast_math: bool = False) -> torch.Tensor:
    """Triton-optimized Swish activation."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose optimal block size based on input size
    if n_elements <= 4096:
        BLOCK_SIZE = 128
    elif n_elements <= 65536:
        BLOCK_SIZE = 256
    elif n_elements <= 262144:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024  # Max threads per block
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    if fast_math:
        swish_fast_kernel[grid](x, output, n_elements, 
                               BLOCK_SIZE=BLOCK_SIZE, USE_FAST_MATH=True)
    else:
        swish_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using Triton kernels.
    """
    def __init__(self, use_fast_math: bool = False):
        super(ModelNew, self).__init__()
        self.use_fast_math = use_fast_math
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # Reshape to 1D for kernel processing if not already contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        return triton_swish(x, self.use_fast_math)
