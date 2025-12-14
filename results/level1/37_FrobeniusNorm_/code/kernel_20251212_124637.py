import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fro_norm_kernel(
    x_ptr,
    norm_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    squares = x * x
    
    # Reduce squares within block
    block_sum = tl.sum(squares, axis=0)
    
    # Atomic add to global norm
    tl.atomic_add(norm_ptr, block_sum)


@triton.jit
def divide_kernel(
    x_ptr,
    norm_val,  # Changed from norm to norm_val to make it clear it's a scalar
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x / norm_val  # Now dividing by scalar value, not pointer
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_fro_norm_normalize(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    
    # Flatten input for kernel processing
    x_flat = x.reshape(-1)
    output_flat = output.reshape(-1)
    n_elements = x_flat.numel()
    
    # Allocate norm tensor on GPU for atomic reduction
    norm_tensor = torch.zeros(1, device=x.device, dtype=x.dtype)
    
    # Use large block size for maximum occupancy
    BLOCK_SIZE = 1024
    
    # Compute sum of squares
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fro_norm_kernel[grid](x_flat, norm_tensor, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute sqrt on GPU and get scalar value
    norm_val = torch.sqrt(norm_tensor[0])
    
    # Avoid division by zero
    norm_val = torch.clamp(norm_val, min=1e-8)
    
    # Divide all elements by norm scalar value
    divide_kernel[grid](x_flat, norm_val, output_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_fro_norm_normalize(x)
