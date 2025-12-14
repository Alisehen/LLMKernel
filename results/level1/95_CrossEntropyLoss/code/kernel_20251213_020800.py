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
    """Optimized cross-entropy kernel with better grid parallelism."""
    # 2D grid layout: (batch_groups, class_groups)
    pid_batch = tl.program_id(0)
    pid_class = tl.program_id(1)
    
    batch_groups = tl.num_programs(0)
    class_groups = tl.num_programs(1)
    
    # Each batch group processes BLOCK_BATCH elements
    BLOCK_BATCH = 16  # Fixed, tuned for 4090
    batch_start = pid_batch * BLOCK_BATCH
    batch_end = tl.minimum(batch_start + BLOCK_BATCH, batch_size)
    
    # Each class group processes BLOCK_SIZE classes
    class_start = pid_class * BLOCK_SIZE
    class_end = tl.minimum(class_start + BLOCK_SIZE, num_classes)
    
    # Initialize local accumulators for this tile
    row_max = tl.full((BLOCK_BATCH,), -float('inf'), dtype=tl.float32)
    row_sum_exp = tl.full((BLOCK_BATCH,), 0.0, dtype=tl.float32)
    
    # First pass: find maximum for each row in tile
    for offset in range(class_start, class_end):
        col_idx = offset
        mask = col_idx < num_classes
        
        # Load predictions for all batch elements in this group at this class
        preds = tl.load(
            predictions_ptr + batch_start * num_classes + col_idx,
            mask=mask & (batch_start + tl.arange(0, BLOCK_BATCH) < batch_size),
            other=-float('inf')
        )
        row_max = tl.maximum(row_max, preds)
    
    # Second pass: compute sum of exp(x - max)
    for offset in range(class_start, class_end):
        col_idx = offset
        mask = col_idx < num_classes
        
        # Load predictions again
        preds = tl.load(
            predictions_ptr + batch_start * num_classes + col_idx,
            mask=mask & (batch_start + tl.arange(0, BLOCK_BATCH) < batch_size),
            other=-float('inf')
        )
        
        # Compute exp(x - max) safely
        x_stable = preds - row_max
        exp_val = tl.exp(x_stable)
        row_sum_exp += exp_val
    
    # Reduce across class groups using atomic operations
    # Store intermediate results in global memory for final reduction
    if pid_class == 0:
        # Load targets for this batch group
        batch_indices = batch_start + tl.arange(0, BLOCK_BATCH)
        batch_mask = batch_indices < batch_size
        target_indices = tl.load(targets_ptr + batch_indices, mask=batch_mask, other=0)
        
        # Compute final loss for each valid batch element
        for i in range(BLOCK_BATCH):
            if batch_mask[i]:
                batch_idx = batch_start + i
                target_idx = target_indices[i]
                
                # Load target prediction
                target_pred = tl.load(predictions_ptr + batch_idx * num_classes + target_idx)
                
                # Compute log_softmax with numerical stability
                log_sum_exp = tl.log(row_sum_exp[i] + 1e-12)
                log_softmax_target = target_pred - row_max[i] - log_sum_exp
                
                # Cross-entropy loss
                loss = -log_softmax_target
                loss_normalized = loss / batch_size
                
                # Atomic add to global loss
                tl.atomic_add(loss_ptr, loss_normalized)

def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused cross entropy loss implementation using Triton."""
    batch_size, num_classes = predictions.shape[0], predictions.shape[1]
    
    # Allocate output (single scalar for loss)
    loss = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Choose optimal block size for class dimension
    BLOCK_SIZE = 256  # Tuned for 4090 memory bandwidth and SM count
    
    # Calculate grid: batch_groups × class_groups
    BLOCK_BATCH = 16  # Each block processes 16 batch elements
    batch_groups = triton.cdiv(batch_size, BLOCK_BATCH)
    class_groups = triton.cdiv(num_classes, BLOCK_SIZE)
    
    # Launch with 2D grid for better parallelism
    grid = (batch_groups, class_groups)
    
    cross_entropy_kernel[grid](
        predictions,
        targets,
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
