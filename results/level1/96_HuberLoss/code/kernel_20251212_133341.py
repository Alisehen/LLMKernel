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
    
    # Compute block sum using tree reduction
    # Initialize reduction register
    block_sum = 0.0
    # Tree reduction using tl.associative_scan with built-in 'add' operation
    # First, ensure loss values are contiguous for reduction
    loss_vec = tl.where(mask, loss, 0.0)
    # Perform associative scan with 'add' operator
    scanned = tl.associative_scan(loss_vec, axis=0, combine_fn='add')
    # Last element contains sum of all elements
    block_sum = scanned[-1]
    
    # Only the last thread in block writes the partial sum
    last_thread_mask = tl.arange(0, BLOCK_SIZE) == (BLOCK_SIZE - 1)
    write_mask = mask & last_thread_mask
    tl.atomic_add(output_ptr + pid, block_sum, mask=write_mask)

@triton.jit
def smooth_l1_loss_final_kernel(
    partial_sums_ptr,
    output_ptr,
    n_partial_sums,
    n_elements_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partial_sums
    
    # Only first block handles final reduction
    if pid == 0:
        partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        # Tree reduction using associative scan
        scanned = tl.associative_scan(partial_sums, axis=0, combine_fn='add')
        total_sum = scanned[-1]
        
        # Load total elements and compute final loss
        n_elements = tl.load(n_elements_ptr)
        final_loss = total_sum / tl.where(n_elements > 0, n_elements, 1.0)
        tl.store(output_ptr, final_loss)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    assert predictions.is_contiguous() and targets.is_contiguous(), "Inputs must be contiguous"
    
    n_elements = predictions.numel()
    
    # Use optimal block size based on tensor size
    BLOCK_SIZE = 1024 if n_elements >= 2**20 else 512
    
    # Allocate partial sums array (one per block)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    n_partial_sums = grid[0]
    
    # Use separate tensors for partial sums and element count
    partial_sums = torch.zeros(n_partial_sums, device=predictions.device, dtype=predictions.dtype)
    n_elements_tensor = torch.tensor([n_elements], device=predictions.device, dtype=predictions.dtype)
    
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
        n_elements_tensor,
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
