import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mse_kernel_optimized(
    pred_ptr,
    target_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized MSE kernel with correct reduction implementation.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Each thread processes BLOCK_SIZE elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    preds = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(target_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared differences
    diff = preds - targets
    squared = diff * diff
    
    # Apply mask for valid elements
    squared_masked = tl.where(mask, squared, 0.0)
    
    # Reduce within thread using efficient tree reduction
    # Start with vector of BLOCK_SIZE elements
    reduced = squared_masked
    
    # Tree reduction: BLOCK_SIZE must be power of 2
    s = BLOCK_SIZE // 2
    while s > 0:
        # Create shifted version
        shifted = tl.where(
            tl.arange(0, BLOCK_SIZE) + s < BLOCK_SIZE,
            tl.load(tl.make_block_ptr(
                reduced,
                shape=(BLOCK_SIZE,),
                strides=(1,),
                offsets=(0,),
                block_shape=(BLOCK_SIZE,),
                order=(0,)
            ) + s, boundary_check=(0,), padding_option="zero"),
            0.0
        )
        
        # Add elements
        reduced = reduced + shifted
        
        # Halve the stride
        s = s // 2
    
    # First element now contains the sum for this block
    block_sum = reduced[0]
    
    # Store block sum
    tl.store(output_ptr + pid, block_sum)

@triton.jit
def reduce_kernel(
    block_sums_ptr,
    output_ptr,
    num_blocks,
    n_elements,
    REDUCE_SIZE: tl.constexpr
):
    """
    Efficient reduction kernel for block sums.
    Uses parallel reduction within thread blocks.
    """
    pid = tl.program_id(axis=0)
    
    # Each thread block processes REDUCE_SIZE block sums
    start_idx = pid * REDUCE_SIZE
    offsets = start_idx + tl.arange(0, REDUCE_SIZE)
    mask = offsets < num_blocks
    
    # Load block sums
    sums = tl.load(block_sums_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce within thread block using tree reduction
    # First, each thread reduces its chunk
    thread_sum = tl.sum(sums)
    
    # Use shared memory for block reduction
    # In Triton, we simulate shared memory with a tensor
    shmem = tl.full((REDUCE_SIZE,), 0.0, dtype=tl.float32)
    thread_id = tl.arange(0, REDUCE_SIZE)
    
    # Store thread sums to shared memory
    shmem = tl.where(thread_id == 0, thread_sum, shmem)
    
    # Tree reduction in shared memory
    offset = REDUCE_SIZE // 2
    while offset > 0:
        # Load from other threads
        other_val = tl.where(
            thread_id + offset < REDUCE_SIZE,
            tl.load(tl.make_block_ptr(
                shmem,
                shape=(REDUCE_SIZE,),
                strides=(1,),
                offsets=(0,),
                block_shape=(REDUCE_SIZE,),
                order=(0,)
            ) + thread_id + offset, boundary_check=(0,), padding_option="zero"),
            0.0
        )
        
        # Add and store back
        shmem = tl.where(thread_id < offset, shmem + other_val, shmem)
        offset = offset // 2
    
    # Thread 0 writes the result
    if thread_id[0] == 0:
        block_total = shmem[0]
        tl.store(output_ptr + pid, block_total)

def triton_mse_optimized(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Optimized MSE computation with correct Triton implementation.
    """
    # Ensure tensors are contiguous
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Flatten tensors for 1D processing
    original_shape = predictions.shape
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    n_elements = predictions.numel()
    
    # Configuration
    if n_elements <= 16384:
        BLOCK_SIZE = 256
    elif n_elements <= 1048576:  # 1M elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Ensure BLOCK_SIZE is power of 2 and <= 1024
    BLOCK_SIZE = min(1024, 1 << (BLOCK_SIZE - 1).bit_length())
    
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensors
    device = predictions.device
    dtype = predictions.dtype
    block_sums = torch.zeros(num_blocks, device=device, dtype=dtype)
    output = torch.zeros(1, device=device, dtype=dtype)
    
    # Launch main MSE kernel
    grid = (num_blocks,)
    mse_kernel_optimized[grid](
        predictions, targets, block_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 256 else 8,
        num_stages=1
    )
    
    # If we have multiple blocks, reduce them
    if num_blocks > 1:
        # Use tree reduction for block sums
        REDUCE_SIZE = 1024
        reduce_blocks = triton.cdiv(num_blocks, REDUCE_SIZE)
        
        # Temporary storage for partial reductions
        partial_sums = torch.zeros(reduce_blocks, device=device, dtype=dtype)
        
        reduce_kernel[(reduce_blocks,)](
            block_sums, partial_sums,
            num_blocks, n_elements,
            REDUCE_SIZE=REDUCE_SIZE,
            num_warps=8,
            num_stages=1
        )
        
        # Final reduction on CPU (small tensor)
        total_sum = partial_sums.sum().item()
    else:
        total_sum = block_sums[0].item()
    
    # Compute final MSE
    output[0] = total_sum / n_elements
    return output

class ModelNew(nn.Module):
    """
    Optimized model that computes Mean Squared Error loss using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, predictions, targets):
        return triton_mse_optimized(predictions, targets)
