import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    batch_size,
    inner_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Compute row (batch index)
    row_idx = pid
    if row_idx >= batch_size:
        return
    
    # Initialize accumulator for this row
    row_sum = 0.0
    
    # Process inner dimension in blocks
    for block_start in range(0, inner_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < inner_dim
        
        # Calculate base pointers for this row
        pred_row_ptr = predictions_ptr + row_idx * inner_dim
        target_row_ptr = targets_ptr + row_idx * inner_dim
        
        # Load predictions and targets for this block
        pred = tl.load(pred_row_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_row_ptr + offsets, mask=mask, other=0.0)
        
        # Avoid log(0) by using epsilon
        epsilon = 1e-12
        safe_pred = tl.maximum(pred, epsilon)
        safe_target = tl.maximum(target, epsilon)
        
        # Compute log(predictions) - log(targets) = log(predictions/targets)
        # Using exp-based formulation: log(x) - log(y) = log(x/y)
        ratio = safe_pred / safe_target
        log_ratio = tl.log(ratio)
        
        # KL divergence: target * (log(target) - log(prediction))
        # = -target * (log(prediction) - log(target))
        kl_div = -target * log_ratio
        
        # Accumulate for this row
        row_sum += tl.sum(kl_div, axis=0)
    
    # Store row sum
    row_output_ptr = output_ptr + row_idx
    tl.store(row_output_ptr, row_sum)

@triton.jit
def batch_mean_kernel(
    row_sums_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load row sums
    row_sums = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Sum reduction using tl.sum
    if pid == 0:
        total_sum = tl.sum(row_sums, axis=0)
        mean = total_sum / batch_size
        tl.store(output_ptr, mean)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    inner_dim = predictions.numel() // batch_size
    
    # Allocate intermediate tensor for row sums
    row_sums = torch.empty(batch_size, dtype=predictions.dtype, device=predictions.device)
    
    # Launch KL divergence kernel
    BLOCK_SIZE = 1024
    grid = (batch_size,)
    kl_div_kernel[grid](
        predictions, targets, row_sums,
        predictions.numel(), batch_size, inner_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch batch mean kernel
    output = torch.empty(1, dtype=predictions.dtype, device=predictions.device)
    grid_mean = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE']),)
    batch_mean_kernel[grid_mean](
        row_sums, output, batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)
