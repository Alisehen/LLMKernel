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
    
    # Compute block sum using efficient tree reduction
    # First, mask out invalid elements
    loss_vec = tl.where(mask, loss, 0.0)
    
    # Tree reduction using tl.reduce
    block_sum = tl.sum(loss_vec, axis=0)
    
    # Store partial sum using atomic add
    if tl.program_id(axis=0) == 0:
        # Only thread 0 in each block accumulates the sum
        tl.atomic_add(output_ptr, block_sum)

@triton.jit
def smooth_l1_loss_final_kernel(
    partial_sum_ptr,
    n_elements,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple final kernel - just divide by n_elements
    # (assuming partial_sum_ptr already contains the total sum)
    if tl.program_id(axis=0) == 0:
        total_sum = tl.load(partial_sum_ptr)
        final_loss = total_sum / tl.where(n_elements > 0, n_elements, 1.0)
        tl.store(output_ptr, final_loss)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.is_contiguous() and targets.is_contiguous(), "Inputs must be contiguous"
    
    n_elements = predictions.numel()
    
    # Use optimal block size
    BLOCK_SIZE = 1024 if n_elements >= 2**20 else 512
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate output tensors
    partial_sum = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Phase 1: Compute partial sums
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        partial_sum,
        n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Phase 2: Final division
    smooth_l1_loss_final_kernel[1](
        partial_sum,
        n_elements,
        output,
        BLOCK_SIZE=1,
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
