import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 16, 'num_stages': 3}),
        triton.Config({'num_warps': 16, 'num_stages': 4}),
        triton.Config({'num_warps': 32, 'num_stages': 2}),
        triton.Config({'num_warps': 32, 'num_stages': 3}),
        triton.Config({'num_warps': 32, 'num_stages': 4}),
    ],
    key=['n_elements', 'BLOCK_SIZE'],
)
@triton.jit
def mse_kernel_optimized(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized MSE kernel with correct reduction using tl.sum.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Each thread processes BLOCK_SIZE elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared differences
    diff = preds - targets
    squared = diff * diff
    
    # Apply mask for valid elements and sum within block
    squared_masked = tl.where(mask, squared, 0.0)
    block_sum = tl.sum(squared_masked)
    
    # Store block sum
    tl.store(output_ptr + pid, block_sum)

def triton_mse_optimized(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Optimized MSE computation with correct Triton implementation.
    """
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Flatten tensors for 1D processing
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    n_elements = predictions.numel()
    
    # Configuration - use powers of 2 for block sizes
    if n_elements <= 16384:
        BLOCK_SIZE = 256
    elif n_elements <= 1048576:  # 1M elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Ensure BLOCK_SIZE is power of 2 and <= 1024
    BLOCK_SIZE = min(1024, 1 << (BLOCK_SIZE.bit_length() - 1))
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensors
    device = predictions.device
    dtype = predictions.dtype
    block_sums = torch.zeros(num_blocks, device=device, dtype=dtype)
    
    # Launch main MSE kernel with autotuned parameters
    grid = (num_blocks,)
    mse_kernel_optimized[grid](
        predictions, targets, block_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Final reduction on GPU (single operation, efficient)
    total_sum = block_sums.sum()
    output = total_sum / n_elements
    
    return output.view(1)

class ModelNew(nn.Module):
    """
    Optimized model that computes Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        return triton_mse_optimized(predictions, targets)
