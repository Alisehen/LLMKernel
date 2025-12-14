import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    inner_dim,
    BLOCK_INNER: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)
    
    # Process multiple batch elements per thread block
    batch_start = pid_batch * BLOCK_BATCH
    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    
    # Process multiple inner dimension elements per thread
    inner_start = pid_inner * BLOCK_INNER
    inner_offsets = inner_start + tl.arange(0, BLOCK_INNER)
    inner_mask = inner_offsets < inner_dim
    
    # Create 2D index arrays for efficient memory access
    batch_idx = batch_offsets[:, None]  # Shape: (BLOCK_BATCH, 1)
    inner_idx = inner_offsets[None, :]  # Shape: (1, BLOCK_INNER)
    
    # Compute pointers for coalesced 2D memory access
    pred_ptrs = predictions_ptr + batch_idx * inner_dim + inner_idx
    target_ptrs = targets_ptr + batch_idx * inner_dim + inner_idx
    
    # Combined mask for valid elements
    mask = batch_mask[:, None] & inner_mask[None, :]
    
    # Load data blocks
    pred_block = tl.load(pred_ptrs, mask=mask, other=0.0)
    target_block = tl.load(target_ptrs, mask=mask, other=0.0)
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-12
    safe_pred = tl.maximum(pred_block, epsilon)
    safe_target = tl.maximum(target_block, epsilon)
    
    # Compute KL divergence: target * (log(target) - log(prediction))
    kl_div = target_block * (tl.log(safe_target) - tl.log(safe_pred))
    
    # Reduce within the block across inner dimension
    acc = tl.sum(kl_div, axis=1)
    
    # Atomic add to accumulate partial sums for each batch element
    output_ptrs = output_ptr + batch_offsets
    tl.atomic_add(output_ptrs, acc, mask=batch_mask)

@triton.jit
def final_mean_kernel(
    row_sums_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load row sums
    row_sums = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Parallel reduction
    sum_val = tl.sum(row_sums, axis=0)
    
    # Atomic add to accumulate total sum
    tl.atomic_add(output_ptr, sum_val)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    inner_dim = predictions.numel() // batch_size
    
    # Allocate output tensors
    row_sums = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Optimized block sizes for V100/A100
    BLOCK_BATCH = 32
    BLOCK_INNER = 256
    
    # 2D grid: batch dimension × inner dimension blocks
    grid_batch = triton.cdiv(batch_size, BLOCK_BATCH)
    grid_inner = triton.cdiv(inner_dim, BLOCK_INNER)
    
    # Autotune configurations for kl_div_kernel
    autotune_configs = [
        {"num_warps": 4, "num_stages": 3},
        {"num_warps": 8, "num_stages": 2},
        {"num_warps": 8, "num_stages": 3},  # Original config
        {"num_warps": 8, "num_stages": 4},
        {"num_warps": 16, "num_stages": 3},
        {"num_warps": 16, "num_stages": 4},
    ]
    
    # Find fastest configuration
    best_time = float('inf')
    best_config = None
    
    for config in autotune_configs:
        # Warm-up
        kl_div_kernel[(grid_batch, grid_inner)](
            predictions, targets, row_sums,
            batch_size, inner_dim,
            BLOCK_INNER=BLOCK_INNER,
            BLOCK_BATCH=BLOCK_BATCH,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"]
        )
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        kl_div_kernel[(grid_batch, grid_inner)](
            predictions, targets, row_sums,
            batch_size, inner_dim,
            BLOCK_INNER=BLOCK_INNER,
            BLOCK_BATCH=BLOCK_BATCH,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"]
        )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        
        if elapsed_time < best_time:
            best_time = elapsed_time
            best_config = config
    
    # Reset row_sums for final run
    row_sums.zero_()
    
    # Launch KL divergence kernel with best config
    kl_div_kernel[(grid_batch, grid_inner)](
        predictions, targets, row_sums,
        batch_size, inner_dim,
        BLOCK_INNER=BLOCK_INNER,
        BLOCK_BATCH=BLOCK_BATCH,
        num_warps=best_config["num_warps"],
        num_stages=best_config["num_stages"]
    )
    
    # Launch final mean reduction (optimized for small grid)
    BLOCK_REDUCE = 1024
    grid_reduce = triton.cdiv(batch_size, BLOCK_REDUCE)
    
    final_mean_kernel[(grid_reduce,)](
        row_sums, output, batch_size,
        BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=4,
        num_stages=1
    )
    
    # Final division by batch_size for batchmean reduction
    output.div_(batch_size)
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div(predictions, targets)
