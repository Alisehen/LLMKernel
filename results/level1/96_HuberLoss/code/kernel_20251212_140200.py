import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Compute offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    preds = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)
    
    # Compute smooth L1 loss
    diff = preds - targets
    abs_diff = tl.abs(diff)
    
    # Conditional operations for Huber loss
    smooth_loss = 0.5 * diff * diff / beta
    linear_loss = abs_diff - 0.5 * beta
    is_small = abs_diff < beta
    loss = tl.where(is_small, smooth_loss, linear_loss)
    
    # Mask invalid elements
    loss = tl.where(mask, loss, 0.0)
    
    # Reduce within the block using tl.sum
    block_sum = tl.sum(loss)
    
    # Store block sum
    tl.store(output_ptr + pid, block_sum)

@triton.jit
def reduce_sum_kernel(
    partial_sum_ptr,
    n_blocks,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    
    # Load partial sums
    partial_sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce within the block using tl.sum
    block_sum = tl.sum(partial_sums)
    
    # Store block sum for final reduction
    tl.store(output_ptr + pid, block_sum)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    device = predictions.device
    dtype = predictions.dtype
    
    # Autotuned configuration based on problem size
    if n_elements >= 1048576:  # 1M+
        BLOCK_SIZE = 1024
        num_stages = 3
    elif n_elements >= 262144:  # 256K+
        BLOCK_SIZE = 512
        num_stages = 4
    else:
        BLOCK_SIZE = 256
        num_stages = 4
    
    # Calculate grid for first kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate partial sums array
    n_blocks = grid[0]
    partial_sums = torch.zeros(n_blocks, device=device, dtype=dtype)
    
    # Phase 1: Compute block-wise sums
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        partial_sums,
        n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Phase 2: Reduce partial sums using tree reduction
    total_sum = torch.zeros(1, device=device, dtype=dtype)
    
    if n_blocks == 1:
        total_sum = partial_sums[0].view(1)
    else:
        # Tree reduction: keep reducing until we get a single value
        current_input = partial_sums
        current_size = n_blocks
        
        while current_size > 1:
            next_grid = (triton.cdiv(current_size, BLOCK_SIZE),)
            next_size = next_grid[0]
            next_partial = torch.zeros(next_size, device=device, dtype=dtype)
            
            reduce_sum_kernel[next_grid](
                current_input,
                current_size,
                next_partial,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            current_input = next_partial
            current_size = next_size
        
        total_sum = current_input[0].view(1)
    
    # Compute mean loss
    return total_sum / float(n_elements)

class ModelNew(nn.Module):
    """
    Optimized model for Smooth L1 (Huber) Loss computation.
    Features efficient block-wise reduction and parallel computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)
