import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 2, 'num_stages': 3}, num_stages=3, num_warps=2),
        triton.Config({'num_warps': 4, 'num_stages': 2}, num_stages=2, num_warps=4),
        triton.Config({'num_warps': 4, 'num_stages': 3}, num_stages=3, num_warps=4),
        triton.Config({'num_warps': 4, 'num_stages': 4}, num_stages=4, num_warps=4),
        triton.Config({'num_warps': 8, 'num_stages': 3}, num_stages=3, num_warps=8),
    ],
    key=['n_elements', 'BLOCK_SIZE', 'VECTORIZE'],
)
@triton.jit
def fused_activation_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VECTORIZE
    
    # Precompute ranges for better instruction scheduling
    range_block = tl.arange(0, BLOCK_SIZE)
    range_vec = tl.arange(0, VECTORIZE)
    
    # Vectorized offsets with improved memory coalescing
    offsets_base = block_start + range_block[:, None] * VECTORIZE
    offsets = offsets_base + range_vec[None, :]
    mask = offsets < n_elements
    
    # Vectorized load with cache hints for L2 residency
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized fused activation using fast math approximations
    # For x > 20: output ≈ x (tanh(softplus(x)) ≈ 1)
    # For x < -20: output ≈ 0 (tanh(softplus(x)) ≈ 0)
    # For moderate values: use optimized approximations
    
    # Fast softplus approximation: log1p(exp(x))
    # Use piecewise approximation for better numerical stability
    exp_x = tl.exp(x)
    exp_neg_x = tl.exp(-x)
    
    # When x > 0: softplus = x + log1p(exp_neg_x)
    # When x <= 0: softplus = log1p(exp_x)
    # Use faster approximations for extreme values
    large_positive_mask = x > 20.0
    large_negative_mask = x < -20.0
    moderate_mask = ~(large_positive_mask | large_negative_mask)
    
    # Compute softplus only for moderate values
    exp_x_masked = tl.where(moderate_mask, exp_x, 0.0)
    exp_neg_x_masked = tl.where(moderate_mask, exp_neg_x, 0.0)
    
    # Optimized softplus computation
    log1p_exp_x = tl.where(x > 0, 0.0, tl.log1p(exp_x_masked))
    log1p_exp_neg_x = tl.where(x <= 0, 0.0, tl.log1p(exp_neg_x_masked))
    softplus_x = tl.where(x > 0, x + log1p_exp_neg_x, log1p_exp_x)
    
    # Fast tanh approximation with fewer branches
    # tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    exp_2y = tl.exp(2.0 * softplus_x)
    tanh_y = (exp_2y - 1.0) / (exp_2y + 1.0)
    
    # Combine all cases
    result = tl.where(large_positive_mask, x,
                     tl.where(large_negative_mask, 0.0,
                             x * tanh_y))
    
    # Vectorized store with cache write-back hint
    tl.store(output_ptr + offsets, result, mask=mask, eviction_policy="evict_last")

def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized configuration based on tensor size
    # For Ada Lovelace (RTX 4090) with 128 SMs
    if n_elements < 500000:
        BLOCK_SIZE = 128  # Smaller blocks for better occupancy with small tensors
        VECTORIZE = 8     # Wider vectorization for better memory throughput
    elif n_elements < 2000000:
        BLOCK_SIZE = 256
        VECTORIZE = 4
    elif n_elements < 10000000:
        BLOCK_SIZE = 512
        VECTORIZE = 2
    else:
        BLOCK_SIZE = 1024  # Full occupancy for large tensors
        VECTORIZE = 1
    
    # Calculate grid size with optimal occupancy
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VECTORIZE']),)
    
    # Launch kernel with autotuned parameters
    fused_activation_kernel_optimized[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VECTORIZE=VECTORIZE,
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused activation using optimized Triton kernel
        x = triton_fused_activation(x)
        
        # Apply batch normalization
        x = self.bn(x)
        return x
