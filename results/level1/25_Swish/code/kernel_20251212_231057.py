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
    """Swish activation optimized with thread coarsening."""
    pid = tl.program_id(axis=0)
    block_start = pid * (BLOCK_SIZE * 4)  # Process 4 elements per thread
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load 4 elements per thread with contiguous access
    mask0 = offsets < n_elements
    mask1 = (offsets + BLOCK_SIZE) < n_elements
    mask2 = (offsets + 2 * BLOCK_SIZE) < n_elements
    mask3 = (offsets + 3 * BLOCK_SIZE) < n_elements
    
    # Load input values
    x0 = tl.load(x_ptr + offsets, mask=mask0, other=0.0)
    x1 = tl.load(x_ptr + offsets + BLOCK_SIZE, mask=mask1, other=0.0)
    x2 = tl.load(x_ptr + offsets + 2 * BLOCK_SIZE, mask=mask2, other=0.0)
    x3 = tl.load(x_ptr + offsets + 3 * BLOCK_SIZE, mask=mask3, other=0.0)
    
    # Process 4 elements using vectorized operations
    # Stable sigmoid implementation
    def compute_swish(x):
        abs_x = tl.abs(x)
        exp_neg_abs_x = tl.exp(-abs_x)
        denom = 1.0 + exp_neg_abs_x
        numerator = tl.where(x >= 0, 1.0, exp_neg_abs_x)
        sigmoid_x = numerator / denom
        return x * sigmoid_x
    
    out0 = compute_swish(x0)
    out1 = compute_swish(x1)
    out2 = compute_swish(x2)
    out3 = compute_swish(x3)
    
    # Store results
    tl.store(output_ptr + offsets, out0, mask=mask0)
    tl.store(output_ptr + offsets + BLOCK_SIZE, out1, mask=mask1)
    tl.store(output_ptr + offsets + 2 * BLOCK_SIZE, out2, mask=mask2)
    tl.store(output_ptr + offsets + 3 * BLOCK_SIZE, out3, mask=mask3)

@triton.jit
def swish_fast_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    USE_FAST_MATH: tl.constexpr,
):
    """Fast math Swish with thread coarsening."""
    pid = tl.program_id(axis=0)
    block_start = pid * (BLOCK_SIZE * 4)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Masks for 4 elements per thread
    mask0 = offsets < n_elements
    mask1 = (offsets + BLOCK_SIZE) < n_elements
    mask2 = (offsets + 2 * BLOCK_SIZE) < n_elements
    mask3 = (offsets + 3 * BLOCK_SIZE) < n_elements
    
    # Load input values
    x0 = tl.load(x_ptr + offsets, mask=mask0, other=0.0)
    x1 = tl.load(x_ptr + offsets + BLOCK_SIZE, mask=mask1, other=0.0)
    x2 = tl.load(x_ptr + offsets + 2 * BLOCK_SIZE, mask=mask2, other=0.0)
    x3 = tl.load(x_ptr + offsets + 3 * BLOCK_SIZE, mask=mask3, other=0.0)
    
    if USE_FAST_MATH:
        # Fast approximation for all 4 elements
        def fast_swish(x):
            half_x = x * 0.5
            exp_x = tl.exp(tl.where(half_x > 10.0, 10.0, tl.where(half_x < -10.0, -10.0, half_x)))
            tanh_half_x = (exp_x - 1.0) / (exp_x + 1.0)
            sigmoid_approx = 0.5 + 0.5 * tanh_half_x
            return x * sigmoid_approx
        
        out0 = fast_swish(x0)
        out1 = fast_swish(x1)
        out2 = fast_swish(x2)
        out3 = fast_swish(x3)
    else:
        # Standard stable implementation
        def stable_swish(x):
            abs_x = tl.abs(x)
            exp_neg_abs_x = tl.exp(-abs_x)
            denom = 1.0 + exp_neg_abs_x
            numerator = tl.where(x >= 0, 1.0, exp_neg_abs_x)
            sigmoid_x = numerator / denom
            return x * sigmoid_x
        
        out0 = stable_swish(x0)
        out1 = stable_swish(x1)
        out2 = stable_swish(x2)
        out3 = stable_swish(x3)
    
    # Store results
    tl.store(output_ptr + offsets, out0, mask=mask0)
    tl.store(output_ptr + offsets + BLOCK_SIZE, out1, mask=mask1)
    tl.store(output_ptr + offsets + 2 * BLOCK_SIZE, out2, mask=mask2)
    tl.store(output_ptr + offsets + 3 * BLOCK_SIZE, out3, mask=mask3)

def triton_swish(x: torch.Tensor, fast_math: bool = False) -> torch.Tensor:
    """Triton-optimized Swish activation with optimized grid layout."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Fixed optimal configuration for Ada Lovelace (4090)
    # 256 threads per block × 4 elements per thread = 1024 elements per block
    BLOCK_SIZE = 256
    
    # Reduced grid size: process 4× more elements per block
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 4),)
    
    if fast_math:
        swish_fast_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_FAST_MATH=True,
            num_warps=8,  # 256 threads / 32 = 8 warps
            num_stages=4
        )
    else:
        swish_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=4
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs Swish activation using optimized Triton kernels.
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
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        
        return triton_swish(x, self.use_fast_math)
