import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hinge_loss_stage1_kernel(
    predictions_ptr,
    targets_ptr,
    partial_sums_ptr,
    partial_counts_ptr,
    n_elements,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 1: Compute clamped loss and partial sums per block.
    Each block processes a contiguous chunk of flattened predictions.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute row and column indices
    row_idx = offsets // n_cols
    col_idx = offsets % n_cols

    # Load predictions and corresponding targets
    preds = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + row_idx, mask=mask)

    # Compute hinge loss: max(0, 1 - predictions * targets)
    product = preds * targets
    loss = 1.0 - product
    clamped_loss = tl.where(loss > 0.0, loss, 0.0)

    # Compute partial sum and count for this block
    # Fix: tl.sum doesn't accept mask parameter, use where to zero out masked values
    clamped_loss_masked = tl.where(mask, clamped_loss, 0.0)
    block_sum = tl.sum(clamped_loss_masked)
    
    # Convert mask to float for counting
    mask_float = tl.where(mask, 1.0, 0.0)
    block_count = tl.sum(mask_float)

    tl.store(partial_sums_ptr + pid, block_sum)
    tl.store(partial_counts_ptr + pid, block_count)


@triton.jit
def reduce_partials_kernel(
    partial_sums_ptr,
    partial_counts_ptr,
    output_sums_ptr,
    output_counts_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduction kernel: Reduce multiple partial sums to fewer partials.
    Each block reduces BLOCK_SIZE elements and writes one output.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials

    # Load partial sums and counts
    partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    partial_counts = tl.load(partial_counts_ptr + offsets, mask=mask, other=0.0)

    # Reduce within the block
    sum_val = tl.sum(partial_sums)
    count_val = tl.sum(partial_counts)

    # Write reduced values
    tl.store(output_sums_ptr + pid, sum_val)
    tl.store(output_counts_ptr + pid, count_val)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of Hinge Loss with efficient multi-stage reduction.
    """
    # Ensure tensors are on GPU and contiguous
    predictions = predictions.contiguous().cuda()
    targets = targets.contiguous().cuda()
    
    n_rows, n_cols = predictions.shape
    n_elements = predictions.numel()
    
    # Allocate output tensor
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Stage 1: Compute partial sums and counts
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_sums = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
    partial_counts = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
    
    grid1 = (num_blocks,)
    hinge_loss_stage1_kernel[grid1](
        predictions, targets, partial_sums, partial_counts,
        n_elements, n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Stage 2+: Recursive reduction until we have 1 value
    current_sums = partial_sums
    current_counts = partial_counts
    current_n = num_blocks
    
    REDUCE_BLOCK_SIZE = 1024
    while current_n > 1:
        next_n = triton.cdiv(current_n, REDUCE_BLOCK_SIZE)
        next_sums = torch.zeros(next_n, device=predictions.device, dtype=predictions.dtype)
        next_counts = torch.zeros(next_n, device=predictions.device, dtype=predictions.dtype)
        
        grid_reduce = (next_n,)
        reduce_partials_kernel[grid_reduce](
            current_sums, current_counts, next_sums, next_counts,
            current_n,
            BLOCK_SIZE=REDUCE_BLOCK_SIZE,
        )
        
        current_sums = next_sums
        current_counts = next_counts
        current_n = next_n
    
    # Final division
    if current_n > 0:
        final_sum = current_sums[0]
        final_count = current_counts[0]
        output[0] = final_sum / final_count if final_count > 0 else 0.0
    
    return output


class ModelNew(nn.Module):
    """
    Optimized Triton implementation of Hinge Loss model.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)
