import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized cross-entropy kernel that processes one batch element per program.
    Each program computes the full softmax and loss for a single batch element.
    """
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # Base pointer for this batch element's predictions
    row_start = pid * num_classes
    
    # Initialize accumulators
    row_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    row_sum_exp = tl.zeros((1,), dtype=tl.float32)
    
    # Process row in chunks of BLOCK_SIZE
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load chunk of predictions
        x = tl.load(predictions_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Update max
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
    
    # Second pass: compute sum of exp(x - max) for normalization
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load chunk of predictions again
        x = tl.load(predictions_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Compute exp(x - max) and accumulate
        exp_val = tl.exp(x - row_max)
        chunk_sum = tl.sum(exp_val, axis=0)
        row_sum_exp += chunk_sum
    
    # Load target class for this batch element
    target_idx = tl.load(targets_ptr + pid)
    
    # Load the target prediction value
    target_pred = tl.load(predictions_ptr + row_start + target_idx)
    
    # Compute log_softmax for the target class
    log_softmax_target = (target_pred - row_max) - tl.log(row_sum_exp + 1e-12)
    
    # Cross-entropy loss = -log_softmax_target
    loss = -log_softmax_target
    
    # Atomic add to global loss (with mean reduction)
    tl.atomic_add(loss_ptr, loss / batch_size)

def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused cross entropy loss implementation using Triton."""
    batch_size, num_classes = predictions.shape[0], predictions.shape[1]
    
    # Allocate output (single scalar for loss)
    loss = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Choose optimal block size (must be power of 2 and <= 1024)
    BLOCK_SIZE = triton.next_power_of_2(min(num_classes, 1024))
    
    # Launch kernel with one program per batch element
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions,
        targets.to(torch.int32),
        loss,
        batch_size,
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return loss

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.
    Optimized implementation using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)
