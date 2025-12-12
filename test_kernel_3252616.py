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
    loss = tl.where(mask, loss, 0.0)
    
    # Efficient tree reduction
    block_sum = tl.sum(loss, axis=0)
    
    # Atomic add for block sum
    tl.atomic_add(output_ptr, block_sum)

@triton.jit
def smooth_l1_loss_final_kernel(
    partial_sum_ptr,
    n_elements,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    if pid == 0:  # Only first thread block processes
        total_sum = tl.load(partial_sum_ptr)
        # Avoid division by zero
        n_elements_val = n_elements if n_elements > 0 else 1
        final_loss = total_sum / n_elements_val
        tl.store(output_ptr, final_loss)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Use optimal block size based on problem size
    if n_elements >= 1048576:  # 2^20
        BLOCK_SIZE = 1024
    elif n_elements >= 262144:  # 2^18
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid size for first phase
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Allocate output tensors on the same device
    device = predictions.device
    dtype = predictions.dtype
    partial_sum = torch.zeros(1, device=device, dtype=dtype)
    output = torch.zeros(1, device=device, dtype=dtype)
    
    # Phase 1: Compute partial sums
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        partial_sum,
        n_elements,
        beta=beta,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Phase 2: Final division - launch single block
    smooth_l1_loss_final_kernel[(1,)](
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
