import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    row_sums_ptr,
    batch_size,
    inner_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Process multiple rows per thread block for better efficiency
    num_rows_per_block = tl.cdiv(batch_size, tl.num_programs(axis=0))
    row_start = pid * num_rows_per_block
    row_end = min(row_start + num_rows_per_block, batch_size)
    
    for row_idx in range(row_start, row_end):
        row_sum = 0.0
        
        # Process inner dimension with vectorized loads
        for block_start in range(0, inner_dim, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < inner_dim
            
            # Load predictions and targets for this block
            pred = tl.load(predictions_ptr + row_idx * inner_dim + offsets, 
                          mask=mask, other=0.0)
            target = tl.load(targets_ptr + row_idx * inner_dim + offsets, 
                            mask=mask, other=0.0)
            
            # Avoid log(0) by adding epsilon
            epsilon = 1e-12
            safe_pred = tl.maximum(pred, epsilon)
            safe_target = tl.maximum(target, epsilon)
            
            # Compute KL divergence: target * (log(target) - log(prediction))
            # Using the identity: log(target) - log(prediction) = log(target/prediction)
            kl_div = target * (tl.log(safe_target) - tl.log(safe_pred))
            
            # Accumulate for this row
            row_sum += tl.sum(kl_div, axis=0)
        
        # Store row sum
        tl.store(row_sums_ptr + row_idx, row_sum)

@triton.jit
def batch_mean_kernel(
    row_sums_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Use parallel reduction for better performance
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load row sums
    row_sums = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Parallel reduction within the block
    sum_val = tl.sum(row_sums, axis=0)
    
    # Use atomic add for final accumulation across blocks
    # Each block writes its partial sum to output[0]
    if tl.program_id(axis=0) == 0:
        # Initialize output to 0 for first block
        tl.store(output_ptr, 0.0)
    
    # Atomic add to accumulate partial sums
    tl.atomic_add(output_ptr, sum_val)

@triton.jit
def final_div_kernel(
    output_ptr,
    batch_size,
):
    # Final division by batch_size
    total = tl.load(output_ptr)
    final_value = total / batch_size
    tl.store(output_ptr, final_value)

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
    # Use fewer thread blocks but process multiple rows per block
    grid = (triton.cdiv(batch_size, 32),)
    kl_div_kernel[grid](
        predictions, targets, row_sums,
        batch_size, inner_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch batch mean kernel with atomic reduction
    output = torch.zeros(1, dtype=predictions.dtype, device=predictions.device)
    grid_mean = (triton.cdiv(batch_size, BLOCK_SIZE),)
    batch_mean_kernel[grid_mean](
        row_sums, output, batch_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final division by batch_size
    final_div_kernel[1](output, batch_size)
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)
