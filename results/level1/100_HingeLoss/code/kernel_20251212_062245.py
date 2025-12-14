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
    block_sum = tl.sum(clamped_loss, mask=mask)
    block_count = tl.sum(mask.to(tl.float32))

    tl.store(partial_sums_ptr + pid, block_sum)
    tl.store(partial_counts_ptr + pid, block_count)


@triton.jit
def hinge_loss_stage2_kernel(
    partial_sums_ptr,
    partial_counts_ptr,
    output_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Stage 2: Reduce partial sums to compute final mean.
    Uses parallel reduction within a single block.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials

    # Load partial sums and counts
    partial_sums = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
    partial_counts = tl.load(partial_counts_ptr + offsets, mask=mask, other=0.0)

    # Reduce within the block
    sum_val = tl.sum(partial_sums)
    count_val = tl.sum(partial_counts)

    # Store to output if this is the first thread in block
    if pid == 0:
        mean_val = sum_val / count_val
        tl.store(output_ptr, mean_val)


@triton.jit
def hinge_loss_fused_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n_elements,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Single-pass computation with parallel reduction.
    Optimized for large inputs using hierarchical reduction.
    """
    pid = tl.program_id(axis=0)
    
    # Phase 1: Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute row indices and load data
    row_idx = offsets // n_cols
    preds = tl.load(predictions_ptr + offsets, mask=mask)
    targets = tl.load(targets_ptr + row_idx, mask=mask)
    
    # Compute hinge loss
    product = preds * targets
    loss = 1.0 - product
    clamped_loss = tl.where(loss > 0.0, loss, 0.0)
    
    # Phase 2: Parallel reduction within block
    block_sum = tl.sum(clamped_loss, mask=mask)
    block_count = tl.sum(mask.to(tl.float32))
    
    # Store partial results in shared memory for final reduction
    partial_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    partial_count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    sum_offset = tl.arange(0, BLOCK_SIZE)
    partial_sum = tl.where(sum_offset == 0, block_sum, partial_sum)
    partial_count = tl.where(sum_offset == 0, block_count, partial_count)
    
    # Parallel reduction within block
    stride = BLOCK_SIZE // 2
    while stride > 0:
        mask_reduce = sum_offset < stride
        shifted_sum = tl.load(partial_sum + stride, mask=mask_reduce)
        shifted_count = tl.load(partial_count + stride, mask=mask_reduce)
        
        current_sum = tl.load(partial_sum, mask=mask_reduce)
        current_count = tl.load(partial_count, mask=mask_reduce)
        
        new_sum = current_sum + shifted_sum
        new_count = current_count + shifted_count
        
        tl.store(partial_sum, new_sum, mask=mask_reduce)
        tl.store(partial_count, new_count, mask=mask_reduce)
        
        stride //= 2
    
    # Store final result from first block
    if pid == 0:
        total_sum = tl.load(partial_sum)
        total_count = tl.load(partial_count)
        mean_val = total_sum / total_count
        tl.store(output_ptr, mean_val)


def triton_hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of Hinge Loss with fused kernel for optimal performance.
    """
    # Ensure tensors are on GPU and contiguous
    predictions = predictions.contiguous().cuda()
    targets = targets.contiguous().cuda()
    
    n_rows, n_cols = predictions.shape
    n_elements = predictions.numel()
    
    # Allocate output tensor
    output = torch.zeros(1, device=predictions.device, dtype=predictions.dtype)
    
    # Choose optimal block size based on problem size
    if n_elements <= 65536:  # Small problem size
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        hinge_loss_fused_kernel[grid](
            predictions, targets, output,
            n_elements, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:  # Large problem size, use two-stage approach
        BLOCK_SIZE = 1024
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        # Stage 1: Compute partial sums
        partial_sums = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
        partial_counts = torch.zeros(num_blocks, device=predictions.device, dtype=predictions.dtype)
        
        grid1 = (num_blocks,)
        hinge_loss_stage1_kernel[grid1](
            predictions, targets, partial_sums, partial_counts,
            n_elements, n_rows, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Stage 2: Reduce partial sums
        grid2 = (1,)
        hinge_loss_stage2_kernel[grid2](
            partial_sums, partial_counts, output,
            num_blocks,
            BLOCK_SIZE=1024,
        )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized Triton implementation of Hinge Loss model.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return triton_hinge_loss(predictions, targets)
