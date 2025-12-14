import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_forward_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CLASSES: tl.constexpr,
):
    # Program IDs: 2D grid for batch x class blocks
    pid_batch = tl.program_id(0)
    pid_class = tl.program_id(1)
    
    # Batch dimension processing
    batch_offsets = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offsets < batch_size
    
    # Class dimension processing  
    class_offsets = pid_class * BLOCK_SIZE_CLASSES + tl.arange(0, BLOCK_SIZE_CLASSES)
    class_mask = class_offsets < num_classes
    
    # Load predictions in tiled manner
    predictions_block = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_CLASSES), dtype=tl.float32)
    
    for i in range(0, BLOCK_SIZE_BATCH, 1):
        for j in range(0, BLOCK_SIZE_CLASSES, 1):
            batch_idx = batch_offsets[i]
            class_idx = class_offsets[j]
            
            if batch_mask[i] and class_mask[j]:
                pred_ptr = predictions_ptr + batch_idx * num_classes + class_idx
                predictions_block = tl.where(
                    tl.arange(0, BLOCK_SIZE_BATCH)[:, None] == i,
                    tl.where(
                        tl.arange(0, BLOCK_SIZE_CLASSES)[None, :] == j,
                        tl.load(pred_ptr),
                        predictions_block
                    ),
                    predictions_block
                )
    
    # Find max for numerical stability (per batch element)
    max_vals = tl.max(predictions_block, axis=1)
    
    # Compute exp and sum per batch element
    exp_vals = tl.exp(predictions_block - max_vals[:, None])
    sum_exp = tl.sum(exp_vals, axis=1)
    log_sum_exp = tl.log(sum_exp) + max_vals
    
    # Load targets for each batch element
    targets_vals = tl.load(targets_ptr + batch_offsets, mask=batch_mask, other=0)
    
    # Compute target logits with tiled access
    target_logits = tl.zeros((BLOCK_SIZE_BATCH,), dtype=tl.float32)
    
    for i in range(BLOCK_SIZE_BATCH):
        if batch_mask[i]:
            batch_idx = batch_offsets[i]
            target_class = tl.load(targets_ptr + batch_idx)
            
            # Find which class block contains the target
            target_class_block = target_class // BLOCK_SIZE_CLASSES
            target_class_offset = target_class % BLOCK_SIZE_CLASSES
            
            if pid_class == target_class_block:
                pred_ptr = predictions_ptr + batch_idx * num_classes + target_class
                target_logits = tl.where(
                    tl.arange(0, BLOCK_SIZE_BATCH) == i,
                    tl.load(pred_ptr),
                    target_logits
                )
    
    # Compute loss per batch element: -log(exp(x_target - log_sum_exp)) = log_sum_exp - x_target
    batch_loss = log_sum_exp - target_logits
    
    # Reduce across class blocks (only one class block per batch element contains the target)
    loss_accumulator = tl.full((BLOCK_SIZE_BATCH,), 0.0, dtype=tl.float32)
    loss_accumulator = tl.where(class_mask[0:1], batch_loss, 0.0)
    
    # Atomic add to global loss
    for i in range(BLOCK_SIZE_BATCH):
        if batch_mask[i]:
            tl.atomic_add(loss_ptr, loss_accumulator[i] / batch_size)

@triton.jit
def fast_cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    # Single block per batch element for better locality
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # Row offset for this batch element
    row_start = pid * num_classes
    row_end = row_start + num_classes
    
    # Process entire row with vectorized loads
    offsets = tl.arange(0, BLOCK_SIZE)
    row_max = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Find max and compute sum_exp in one pass
    for i in range(0, num_classes, BLOCK_SIZE):
        col_offsets = i + offsets
        mask = col_offsets < num_classes
        
        # Load predictions for this segment
        x = tl.load(predictions_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
        
        # Update max
        current_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, current_max)
        
        # Compute exp(x - current_max) for sum
        exp_val = tl.exp(x - current_max)
        row_sum += tl.sum(exp_val, axis=0)
    
    # Broadcast final max and sum
    final_max = tl.max(row_max, axis=0)
    final_sum = tl.sum(row_sum, axis=0)
    
    # Load target
    target_idx = tl.load(targets_ptr + pid)
    
    # Load target prediction
    target_pred = tl.load(predictions_ptr + row_start + target_idx)
    
    # Compute log_sum_exp
    log_sum_exp = tl.log(final_sum) + final_max
    
    # Compute loss for this batch element
    loss = log_sum_exp - target_pred
    
    # Atomic add to output (mean reduction)
    tl.atomic_add(loss_ptr, loss / batch_size)

def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused cross entropy loss implementation using Triton."""
    batch_size, num_classes = predictions.shape[0], predictions.shape[1]
    
    # Allocate output (single scalar for loss)
    loss = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Choose optimal block size
    BLOCK_SIZE = 1024  # Maximum threads per block
    
    # Use fast single-block-per-row kernel for better performance
    grid = (triton.cdiv(batch_size, 1),)
    
    fast_cross_entropy_kernel[grid](
        predictions,
        targets.to(torch.int32),  # Convert to int32 for Triton compatibility
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
