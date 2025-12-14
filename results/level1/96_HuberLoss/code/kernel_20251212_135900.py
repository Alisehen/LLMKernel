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
    num_stages: tl.constexpr = 3,
):
    pid = tl.program_id(axis=0)
    
    # Compute offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load predictions and targets
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute smooth L1 loss
    diff = preds - targets
    abs_diff = tl.abs(diff)
    
    # Conditional operations
    is_small = abs_diff < beta
    smooth_loss = 0.5 * diff * diff / beta
    linear_loss = abs_diff - 0.5 * beta
    loss = tl.where(is_small, smooth_loss, linear_loss)
    
    # Only include valid elements in reduction
    loss = tl.where(mask, loss, 0.0)
    
    # Reduce within the block
    block_sum = tl.sum(loss)
    
    # Store block sum to global memory
    block_sum_ptr = output_ptr + pid
    tl.store(block_sum_ptr, block_sum)

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
    
    # Reduce within the block
    block_sum = tl.sum(partial_sums)
    
    # Final reduction across blocks
    if pid == 0:
        total_sum = tl.sum(block_sum)
        tl.store(output_ptr, total_sum)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    device = predictions.device
    dtype = predictions.dtype
    
    # Autotuned configuration
    configs = [
        {"BLOCK_SIZE": 1024, "num_stages": 3},
        {"BLOCK_SIZE": 512, "num_stages": 4},
        {"BLOCK_SIZE": 256, "num_stages": 4},
    ]
    
    # Choose optimal configuration
    if n_elements >= 1048576:  # 1M+
        config = configs[0]
    elif n_elements >= 262144:  # 256K+
        config = configs[1]
    else:
        config = configs[2]
    
    BLOCK_SIZE = config["BLOCK_SIZE"]
    num_stages = config["num_stages"]
    
    # Calculate grid for first kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate partial sums array
    n_blocks = grid[0]
    partial_sums = torch.zeros(n_blocks, device=device, dtype=dtype)
    output = torch.zeros(1, device=device, dtype=dtype)
    
    # Phase 1: Compute block-wise sums
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        partial_sums,
        n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
    )
    
    # Phase 2: Reduce partial sums
    if n_blocks > 1:
        reduce_grid = (triton.cdiv(n_blocks, BLOCK_SIZE),)
        reduce_sum_kernel[reduce_grid](
            partial_sums,
            n_blocks,
            output,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # If only one block, just copy the result
        output = partial_sums[0].view(1)
    
    # Compute mean loss
    n_elements_val = float(n_elements) if n_elements > 0 else 1.0
    return output / n_elements_val

class ModelNew(nn.Module):
    """
    Optimized model for Smooth L1 (Huber) Loss computation.
    Features efficient block-wise reduction and parallel computation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)
