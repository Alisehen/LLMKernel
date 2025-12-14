import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
    ],
    key=['n_elements']
)
@triton.jit
def smooth_l1_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Create block offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data in native precision and convert to float32 for precise computation
    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    diff = preds - targets
    abs_diff = tl.abs(diff)
    
    # Smooth L1 Loss: 0.5 * x^2 / beta if |x| < beta else |x| - 0.5 * beta
    is_small = abs_diff < beta
    smooth_loss = 0.5 * diff * diff / beta
    linear_loss = abs_diff - 0.5 * beta
    
    loss = tl.where(is_small, smooth_loss, linear_loss)
    
    # Efficient reduction using tl.reduce
    block_sum = tl.sum(loss)
    
    # Atomic add to accumulate results across blocks
    tl.atomic_add(output_ptr + tl.zeros((1,), dtype=tl.int32), block_sum)

def triton_smooth_l1_loss(predictions: torch.Tensor, targets: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    assert predictions.shape == targets.shape, "Predictions and targets must have same shape"
    
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    n_elements = predictions.numel()
    
    # Allocate output tensor for reduction (single element)
    device = predictions.device
    output = torch.zeros(1, device=device, dtype=torch.float32)
    
    # Calculate grid size
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    # Launch kernel
    smooth_l1_loss_kernel[grid](
        predictions,
        targets,
        output,
        n_elements,
        beta=beta,
    )
    
    # Normalize by number of elements
    return output / n_elements

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_smooth_l1_loss(predictions, targets)
