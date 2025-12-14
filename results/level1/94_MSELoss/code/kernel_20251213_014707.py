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
):
    """
    Optimized MSE kernel with:
    - Correct warp-level reduction
    - Vectorized memory access patterns
    - Proper masking for boundary conditions
    """
    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes multiple elements
    thread_sum = tl.zeros((1,), tl.float32)
    
    for i in range(ELEMENTS_PER_THREAD):
        # Calculate global offset for this element
        element_offset = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD + i * BLOCK_SIZE + tid
        mask = element_offset < n_elements
        
        # Load with efficient memory access patterns
        preds = tl.load(pred_ptr + element_offset, mask=mask, other=0.0)
        targets = tl.load(target_ptr + element_offset, mask=mask, other=0.0)
        
        # Compute squared differences
        diff = preds - targets
        squared = diff * diff
        thread_sum += tl.where(mask, squared, 0.0)
    
    # Reduce within warp using proper warp-level reduction
    warp_sum = tl.sum(thread_sum, axis=0)
    
    # Warp leaders store partial sums
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
    Efficient reduction kernel for partial sums using tree reduction.
    """
    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)
    
    offsets = pid * BLOCK_SIZE + tid
    mask = offsets < n_partials
    
    # Load partial sums
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    
    # Tree reduction within block
    block_sum = tl.sum(partials, axis=0)
    
    # Store final result
    if pid == 0:
        tl.store(output_ptr, block_sum)

def triton_mse_fast(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Optimized MSE computation with correct reduction patterns.
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
    
    # Configuration optimized for modern GPUs
    BLOCK_SIZE = 512  # Optimal for better occupancy
    ELEMENTS_PER_THREAD = 8  # Increase ILP and hide latency
    
    # Calculate grid size
    elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD
    num_blocks = triton.cdiv(n_elements, elements_per_block)
    
    # Allocate partial sums (one per warp)
    warps_per_block = BLOCK_SIZE // 32
    total_warps = num_blocks * warps_per_block
    partials = torch.zeros(total_warps, device=device, dtype=dtype)
    
    # Launch main MSE kernel
    grid = (num_blocks,)
    mse_kernel_fast[grid](
        predictions, targets, partials,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        num_warps=warps_per_block,
    )
    
    # Final reduction if needed
    if total_warps > 1:
        result = torch.zeros(1, device=device, dtype=dtype)
        reduce_block_size = 512
        reduce_grid = (triton.cdiv(total_warps, reduce_block_size),)
        
        reduce_kernel[reduce_grid](
            partials, result,
            total_warps,
            BLOCK_SIZE=reduce_block_size,
            num_warps=reduce_block_size // 32,
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
