import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fast GELU activation using Triton."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Precompute constants inside kernel
    SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/π)
    GELU_CONST = 0.044715
    
    x3 = x * x * x
    inner = x + GELU_CONST * x3
    scaled = SQRT_2_OVER_PI * inner
    
    # Compute tanh using exp-based formulation (tanh(x) = 1 - 2/(exp(2x) + 1))
    exp_2x = tl.exp(2.0 * scaled)
    tanh_result = 1.0 - 2.0 / (exp_2x + 1.0)
    
    output = 0.5 * x * (1.0 + tanh_result)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def gelu_kernel_fp16(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU for FP16/BF16 with fewer registers."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)  # Convert to FP32 for higher precision
    
    # Precompute constants inside kernel
    SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/π)
    GELU_CONST = 0.044715
    
    # Compute GELU using fused operations
    x3 = x_f32 * x_f32 * x_f32
    inner = x_f32 + GELU_CONST * x3
    scaled = SQRT_2_OVER_PI * inner
    
    # Fast tanh approximation for better throughput
    # tanh(x) ≈ x * (27 + x^2) / (27 + 9 * x^2) for x in [-3, 3]
    # Beyond ±3, tanh(x) ≈ sign(x) * (1 - exp(-2*|x|))
    abs_scaled = tl.abs(scaled)
    
    # For small values use polynomial approximation
    small_mask = abs_scaled <= 3.0
    scaled_sq = scaled * scaled
    tanh_small = scaled * (27.0 + scaled_sq) / (27.0 + 9.0 * scaled_sq)
    
    # For large values use exp-based approximation
    exp_neg_2abs = tl.exp(-2.0 * abs_scaled)
    tanh_large = tl.where(scaled >= 0, 1.0 - exp_neg_2abs, -1.0 + exp_neg_2abs)
    
    tanh_result = tl.where(small_mask, tanh_small, tanh_large)
    
    output = 0.5 * x_f32 * (1.0 + tanh_result)
    tl.store(output_ptr + offsets, output.to(x.dtype), mask=mask)

def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function for GELU activation."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose kernel and block size based on dtype and size
    if x.dtype in [torch.float16, torch.bfloat16]:
        # Use optimized kernel for lower precision
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        gelu_kernel_fp16[grid](x, output, n_elements, BLOCK_SIZE)
    else:
        # Use standard kernel for FP32
        BLOCK_SIZE = 512  # Smaller block for FP32 due to register pressure
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model with Triton GELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation using optimized Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)
