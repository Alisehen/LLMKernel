import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel_fast(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Optimized MSE kernel with:
    - Vectorized memory access
    - Multiple elements per thread for better latency hiding
    - Configurable prefetching stages
    """
    pid = tl.program_id(axis=0)
    
    # Each thread processes ELEMENTS_PER_THREAD elements
    thread_offset = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD
    thread_sum = tl.zeros((1,), tl.float32)
    
    # Process multiple elements per thread
    for i in range(ELEMENTS_PER_THREAD):
        block_offset = thread_offset + i * BLOCK_SIZE
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load with prefetching hint
        preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        targets = tl.load(target_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        
        # Compute squared differences
        diff = preds - targets
        squared = diff * diff
        
        # Mask invalid elements and accumulate
        squared_masked = tl.where(mask, squared, 0.0)
        thread_sum += tl.sum(squared_masked, axis=0)
    
    # Reduce within warp
    warp_sum = tl.sum(thread_sum)
    
    # Store warp sum if this is the first thread in warp
    if tl.program_id(axis=0) % 32 == 0:
        warp_id = tl.program_id(axis=0) // 32
        tl.store(output_ptr + warp_id, warp_sum)

@triton.jit
def reduce_kernel(
    partial_ptr,
    output_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast reduction kernel for partial sums.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    
    # Load partial sums
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce within block
    block_sum = tl.sum(partials)
    
    # Store final result if first thread
    if pid == 0:
        tl.store(output_ptr, block_sum)

def triton_mse_fast(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Optimized MSE computation with:
    - Better memory access patterns
    - Warp-level reduction to minimize global memory writes
    - Configurable number of stages for latency hiding
    """
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Flatten tensors for 1D processing
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    n_elements = predictions.numel()
    device = predictions.device
    dtype = predictions.dtype
    
    # Configuration tuned for Ada Lovelace
    BLOCK_SIZE = 1024  # Maximum threads per block for maximum occupancy
    ELEMENTS_PER_THREAD = 4  # Process 4 elements per thread for better latency hiding
    NUM_STAGES = 3  # Increased for better memory latency hiding
    
    # Calculate grid size - each block processes BLOCK_SIZE * ELEMENTS_PER_THREAD elements
    elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD
    num_blocks = triton.cdiv(n_elements, elements_per_block)
    
    # Allocate partial sums (one per warp)
    warps_per_grid = triton.cdiv(num_blocks, 32)
    partials = torch.zeros(warps_per_grid, device=device, dtype=dtype)
    
    # Launch main MSE kernel
    grid = (num_blocks,)
    mse_kernel_fast[grid](
        predictions, targets, partials,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        NUM_STAGES=NUM_STAGES,
        num_warps=BLOCK_SIZE // 32,
    )
    
    # Final reduction if needed
    if warps_per_grid > 1:
        result = torch.zeros(1, device=device, dtype=dtype)
        reduce_block_size = min(1024, warps_per_grid)
        reduce_grid = (triton.cdiv(warps_per_grid, reduce_block_size),)
        
        reduce_kernel[reduce_grid](
            partials, result,
            warps_per_grid,
            BLOCK_SIZE=reduce_block_size,
            num_warps=reduce_block_size // 32,
            num_stages=2,
        )
        total_sum = result.item()
    else:
        total_sum = partials.item()
    
    # Compute final MSE
    output = total_sum / n_elements
    return torch.tensor([output], device=device, dtype=dtype)

class ModelNew(nn.Module):
    """
    Optimized model that computes Mean Squared Error loss using fast Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        return triton_mse_fast(predictions, targets)
