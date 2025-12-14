import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 3}),
        triton.Config({'num_warps': 4, 'num_stages': 4}),
        triton.Config({'num_warps': 8, 'num_stages': 3}),
        triton.Config({'num_warps': 8, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 2}),
        triton.Config({'num_warps': 16, 'num_stages': 3}),
    ],
    key=['batch_size', 'num_classes'],
)
@triton.jit
def cross_entropy_kernel(
    predictions_ptr,
    targets_ptr,
    loss_ptr,
    batch_size,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Optimized cross-entropy kernel using two-pass reduction."""
    pid = tl.program_id(0)
    if pid >= batch_size:
        return
    
    # Base pointer for this batch element
    row_start = pid * num_classes
    
    # Initialize accumulators as scalars (not tensors)
    row_max = -float('inf')
    row_sum_exp = 0.0
    
    # First pass: find maximum for numerical stability
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load predictions chunk
        x = tl.load(predictions_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
    
    # Second pass: compute sum of exp(x - max)
    for start in range(0, num_classes, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_classes
        
        # Load predictions chunk again
        x = tl.load(predictions_ptr + row_start + offsets, mask=mask, other=-float('inf'))
        
        # Compute exp(x - max) safely
        x_stable = x - row_max
        exp_val = tl.exp(x_stable)
        chunk_sum = tl.sum(exp_val, axis=0)
        row_sum_exp += chunk_sum
    
    # Load target class and its prediction
    target_idx = tl.load(targets_ptr + pid)
    target_pred = tl.load(predictions_ptr + row_start + target_idx)
    
    # Compute log_softmax with numerical stability
    log_sum_exp = tl.log(row_sum_exp + 1e-12)
    log_softmax_target = target_pred - row_max - log_sum_exp
    
    # Cross-entropy loss
    loss = -log_softmax_target
    
    # Atomic add to global loss (don't normalize here)
    tl.atomic_add(loss_ptr, loss)

def triton_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Fused cross entropy loss implementation using Triton."""
    batch_size, num_classes = predictions.shape[0], predictions.shape[1]
    
    # Allocate output (single scalar for loss)
    loss = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Choose optimal block size (power of 2, <= 1024)
    BLOCK_SIZE = triton.next_power_of_2(min(num_classes, 1024))
    
    # Launch kernel with one program per batch element
    grid = (batch_size,)
    
    cross_entropy_kernel[grid](
        predictions,
        targets,
        loss,
        batch_size,
        num_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Normalize by batch size after reduction
    return loss / batch_size

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.
    Optimized implementation using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_cross_entropy(predictions, targets)
