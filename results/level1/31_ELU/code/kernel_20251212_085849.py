import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def elu_kernel_optimized(
    x_ptr,
    output_ptr,
    alpha_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    # 2D grid for better parallelism
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block start position with 2D decomposition
    block_start_m = pid_m * BLOCK_SIZE
    block_start_n = pid_n * ELEMENTS_PER_THREAD
    
    # Create offsets for the entire block
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE)
    offsets_n = block_start_n + tl.arange(0, ELEMENTS_PER_THREAD)
    
    # Create mask for boundary checking
    mask_m = offsets_m < n_elements
    mask_n = offsets_n < ELEMENTS_PER_THREAD  # Always true, but keeps pattern consistent
    
    # Process multiple elements per thread with vectorization
    for i in tl.range(ELEMENTS_PER_THREAD):
        if block_start_n + i < n_elements:  # Check global boundary
            # Calculate global offset
            global_offset = offsets_m + (block_start_n + i) * n_elements
            
            # Load with proper masking
            load_mask = mask_m & (offsets_m < n_elements) & ((block_start_n + i) < n_elements)
            x = tl.load(x_ptr + global_offset, mask=load_mask, other=0.0)
            
            # ELU computation
            exp_x = tl.exp(x)
            elu_x = tl.where(x > 0, x, alpha_val * (exp_x - 1.0))
            
            # Store result
            tl.store(output_ptr + global_offset, elu_x, mask=load_mask)


def triton_elu_optimized(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized configuration based on Ada Lovelace architecture
    # Using smaller blocks but more parallelism via 2D grid
    BLOCK_SIZE = 256
    ELEMENTS_PER_THREAD = 4
    
    # Calculate 2D grid dimensions
    grid_m = triton.cdiv(n_elements, BLOCK_SIZE)
    grid_n = triton.cdiv(n_elements, ELEMENTS_PER_THREAD)
    
    # Use 2D grid for better SM utilization
    grid = (grid_m, grid_n)
    
    # Convert alpha to the same dtype as x for kernel
    alpha_val = torch.tensor(alpha, dtype=x.dtype, device=x.device).item()
    
    elu_kernel_optimized[grid](
        x, 
        output, 
        alpha_val, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu_optimized(x, alpha=self.alpha)
