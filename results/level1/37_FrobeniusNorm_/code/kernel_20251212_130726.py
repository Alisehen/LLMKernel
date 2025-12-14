import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_fro_norm_normalize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr = 3,
):
    """
    Fused kernel that computes Frobenius norm and normalizes in a single pass.
    Uses efficient block reduction without inter-block synchronization issues.
    """
    pid = tl.program_id(axis=0)
    
    # Each thread processes BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with coalesced access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute local sum of squares
    local_sum = tl.sum(x * x)
    
    # Allocate shared memory for block reduction
    shmem = tl.zeros((128,), dtype=tl.float32)
    shmem_offset = tl.program_id(axis=0) % 128
    
    # Store local sum to shared memory
    tl.store(shmem + shmem_offset, local_sum)
    tl.debug_barrier()
    
    # Thread 0 reduces block sums from shared memory
    if tl.program_id(axis=0) == 0:
        # Load all block sums (up to 128 blocks)
        block_sums = tl.load(shmem + tl.arange(0, 128))
        global_sum = tl.sum(block_sums)
        norm_val = tl.sqrt(global_sum) + EPS
        
        # Store norm value in first element of output
        tl.store(output_ptr, norm_val)
    
    tl.debug_barrier()
    
    # All threads read the computed norm value
    norm_val = tl.load(output_ptr)
    
    # Normalize and store results
    normalized = x / norm_val
    tl.store(output_ptr + offsets, normalized, mask=mask)


def triton_fro_norm_normalize(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    
    # Flatten input for kernel processing
    x_flat = x.reshape(-1)
    output_flat = output.reshape(-1)
    n_elements = x_flat.numel()
    
    # Configuration optimized for performance
    BLOCK_SIZE = 1024
    
    # Calculate grid size - ensure we have enough blocks for full utilization
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch fused kernel
    fused_fro_norm_normalize_kernel[(num_blocks,)](
        x_flat,
        output_flat,
        n_elements,
        EPS=1e-8,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=3,
        num_warps=8,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_fro_norm_normalize(x)
