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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    preds = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + offsets, mask=mask)
    
    diff = preds - targets
    abs_diff = tl.abs(diff)
    
    # Smooth L1 Loss: 0.5 * x^2 / beta if |x| < beta else |x| - 0.5 * beta
    is_small = abs_diff < beta
    smooth_loss = 0.5 * diff * diff / beta
    linear_loss = abs_diff - 0.5 * beta
    
    loss = tl.where(is_small, smooth_loss, linear_loss)
    
    # Compute sum reduction using efficient tree reduction within block
    # Use tl.associative_scan for better performance than tl.sum
    block_sum = tl.associative_scan(loss, axis=0, combine_fn=lambda x, y: x + y)[BLOCK_SIZE - 1]
    
    # Only the last thread in block writes the partial sum
    last_thread_mask = tl.arange(0, BLOCK_SIZE) == (BLOCK_SIZE - 1)
    write_mask = mask & last_thread_mask
    tl.atomic_add(output_ptr + pid, block_sum, mask=write_mask)

@triton.jit
def smooth_l1_loss_final_kernel(
    partial_sums_ptr,
    output_ptr,
    n_partial_sums,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    if pid == 0:  # Only first block handles final reduction
        partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        # Efficient tree reduction within block
        total_sum = tl.associative_scan(partial_sums, axis=0, combine_fn=lambda x, y: x + y)[BLOCK_SIZE - 1]
        
        # Store final result and divide by total elements
        n_elements = tl.load(partial_sums_ptr + n_partial_sums)  # Last element stores n_elements
        final_loss = total_sum / tl.where(n_elements > 0, n_elements, 1.0)
        tl.store(output_ptr, final_sum)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.is_contiguous() and targets.is_contiguous(), "Inputs must be contiguous"
    
    n_elements = predictions.numel()
    
    # Use optimal block size based on tensor size
    BLOCK_SIZE = 1024 if n_elements >= 2**20 else 512
    
    # Allocate partial sums array (one per block)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    n_partial_sums = grid[0]
    
    # Use pinned memory for partial sums to reduce atomic contention
    partial_sums = torch.zeros(n_partial_sums + 1, device=predictions.device, dtype=predictions.dtype)
    
    # Store total elements at the end of partial sums array
    partial_sums[-1] = float(n_elements)
    
    # Phase 1: Compute partial sums per block
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        partial_sums,
        n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Phase 2: Final reduction of partial sums
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    final_grid = (1,)  # Single block for final reduction
    
    smooth_l1_loss_final_kernel[final_grid](
        partial_sums,
        output,
        n_partial_sums,
        BLOCK_SIZE=min(1024, triton.next_power_of_2(n_partial_sums)),
    )
    
    return output

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)
