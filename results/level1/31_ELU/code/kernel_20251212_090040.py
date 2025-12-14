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
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ELU computation using Triton's built-in operations
    # elu(x) = x if x > 0 else alpha * (exp(x) - 1)
    elu_x = tl.where(x > 0, x, alpha_val * (tl.exp(x) - 1.0))
    
    # Store result
    tl.store(output_ptr + offsets, elu_x, mask=mask)


def triton_elu_optimized(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized block size for best performance
    # Using large block size to maximize parallelism
    BLOCK_SIZE = 1024
    
    # Calculate 1D grid dimensions
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Convert alpha to the same dtype as x
    if x.dtype == torch.float16:
        alpha_val = torch.tensor(alpha, dtype=torch.float16, device=x.device).item()
    elif x.dtype == torch.bfloat16:
        alpha_val = torch.tensor(alpha, dtype=torch.bfloat16, device=x.device).item()
    else:
        alpha_val = torch.tensor(alpha, dtype=torch.float32, device=x.device).item()
    
    elu_kernel_optimized[grid](
        x,
        output,
        alpha_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_elu_optimized(x, alpha=self.alpha)
