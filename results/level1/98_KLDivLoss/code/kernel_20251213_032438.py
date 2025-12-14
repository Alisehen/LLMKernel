import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kl_div_kernel_optimized(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    inner_dim,
    # Tile sizes for better cache utilization
    BLOCK_BATCH: tl.constexpr,
    TILE_K: tl.constexpr,  # Inner dimension tile for vectorized loads
    USE_TENSOR_CORES: tl.constexpr,
):
    # pid: each thread block processes a chunk of batch elements
    pid = tl.program_id(axis=0)
    
    # Compute batch range for this block
    batch_start = pid * BLOCK_BATCH
    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    
    # Initialize accumulator registers for each batch element
    acc = tl.zeros((BLOCK_BATCH,), dtype=tl.float32)
    
    # Loop over inner dimension in tiles for better cache reuse
    for k in range(0, inner_dim, TILE_K):
        k_offsets = k + tl.arange(0, TILE_K)
        k_mask = k_offsets < inner_dim
        
        # Broadcast indices for 2D access - better coalescing
        batch_idx = tl.reshape(batch_offsets, (BLOCK_BATCH, 1))
        k_idx = tl.reshape(k_offsets, (1, TILE_K))
        
        # Create pointers with proper 2D indexing
        pred_ptrs = predictions_ptr + batch_idx * inner_dim + k_idx
        target_ptrs = targets_ptr + batch_idx * inner_dim + k_idx
        
        # Combined mask for valid elements
        mask = tl.reshape(batch_mask, (BLOCK_BATCH, 1)) & tl.reshape(k_mask, (1, TILE_K))
        
        # Load tile with type promotion for Tensor Cores if enabled
        if USE_TENSOR_CORES:
            pred_tile = tl.load(pred_ptrs, mask=mask, other=0.0).to(tl.float32)
            target_tile = tl.load(target_ptrs, mask=mask, other=0.0).to(tl.float32)
        else:
            pred_tile = tl.load(pred_ptrs, mask=mask, other=0.0)
            target_tile = tl.load(target_ptrs, mask=mask, other=0.0)
        
        # Add epsilon safely (avoiding underflow)
        epsilon = 1e-12
        safe_pred = tl.maximum(pred_tile, epsilon)
        safe_target = tl.maximum(target_tile, epsilon)
        
        # Compute KL divergence for this tile
        log_target = tl.log(safe_target)
        log_pred = tl.log(safe_pred)
        
        # Fused KL computation with reduction across inner dimension
        kl_tile = target_tile * (log_target - log_pred)
        
        # Reduce tile across inner dimension and accumulate
        tile_sum = tl.sum(kl_tile, axis=1)
        acc += tl.where(batch_mask, tile_sum, 0.0)
    
    # Store accumulated results
    output_ptrs = output_ptr + batch_offsets
    tl.store(output_ptrs, acc, mask=batch_mask)

@triton.jit
def fast_mean_reduction(
    row_sums_ptr,
    output_ptr,
    batch_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    
    # Load with coalesced access
    row_sums = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Tree reduction within thread block
    # Using warp shuffle operations for efficient reduction
    sum_val = tl.sum(row_sums, axis=0)
    
    # First thread in block writes the result
    if pid == 0:
        tl.store(output_ptr, sum_val)

def triton_kl_div_optimized(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are contiguous and aligned
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    batch_size = predictions.shape[0]
    inner_dim = predictions.shape[1] if predictions.ndim == 2 else predictions.numel() // batch_size
    
    # Allocate output tensors (pinned memory if available)
    row_sums = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Optimized configuration for Ada Lovelace (4090)
    # Check if Tensor Cores would be beneficial
    use_tensor_cores = (inner_dim >= 512 and 
                       predictions.dtype in [torch.float16, torch.bfloat16])
    
    if use_tensor_cores:
        # Tensor Core optimized configuration
        BLOCK_BATCH = 64  # Increased for better occupancy
        TILE_K = 64  # Optimized tile for memory reuse
    else:
        # Standard configuration for smaller matrices
        BLOCK_BATCH = 128  # Maximize batch processing
        TILE_K = 128  # Larger tile for better cache reuse
    
    # Calculate grid size
    grid_batch = triton.cdiv(batch_size, BLOCK_BATCH)
    
    # Launch optimized KL divergence kernel with appropriate stages
    num_stages = 3 if use_tensor_cores else 2  # Tensor Cores benefit from more stages
    
    kl_div_kernel_optimized[(grid_batch,)](
        predictions, targets, row_sums,
        batch_size, inner_dim,
        BLOCK_BATCH=BLOCK_BATCH,
        TILE_K=TILE_K,
        USE_TENSOR_CORES=use_tensor_cores,
        num_warps=8 if use_tensor_cores else 4,
        num_stages=num_stages
    )
    
    # Launch optimized reduction kernel
    # Use appropriate block size based on batch size
    BLOCK_REDUCE = min(1024, triton.next_power_of_2(batch_size))
    grid_reduce = 1  # Single block for final reduction
    
    fast_mean_reduction[(grid_reduce,)](
        row_sums, output, batch_size,
        BLOCK_SIZE=BLOCK_REDUCE,
        num_warps=4,
        num_stages=1
    )
    
    # Final division by batch_size for batchmean reduction
    result = output[0] / batch_size
    
    return result

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return triton_kl_div_optimized(predictions, targets)
