import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def selu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SELU parameters (precomputed constants)
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946
    
    # Compute SELU: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    pos_mask = x > 0.0
    pos_part = tl.where(pos_mask, x, 0.0)
    
    # exp(x) - 1 for negative values using high-precision computation
    exp_x = tl.exp(x)
    neg_part = ALPHA * (exp_x - 1.0)
    neg_part = tl.where(pos_mask, 0.0, neg_part)
    
    result = SCALE * (pos_part + neg_part)
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def selu_kernel_optimized(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized version with fused operations and fewer conditionals.
    Uses mathematical identity: max(0,x) = x * (x > 0), min(0,x) = x * (x <= 0)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # SELU parameters (precomputed constants)
    ALPHA = 1.6732632423543772848170429916717
    SCALE = 1.0507009873554804934193349852946
    ALPHA_SCALE = ALPHA * SCALE
    NEG_ALPHA_SCALE = -ALPHA * SCALE
    
    # Compute condition once
    pos_cond = x > 0.0
    
    # Compute both branches
    pos_part = x * pos_cond  # x if x > 0, else 0
    
    # For negative branch: scale * alpha * (exp(x) - 1)
    exp_x = tl.exp(x)
    neg_part = ALPHA_SCALE * (exp_x - 1.0)
    neg_part = neg_part * (~pos_cond)  # apply only to negative values
    
    # Final result
    result = SCALE * pos_part + neg_part
    
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_selu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose optimal block size based on tensor size
    if n_elements >= 1048576:  # 1M elements
        BLOCK_SIZE = 1024
    elif n_elements >= 262144:  # 256K elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Use optimized kernel
    selu_kernel_optimized[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor using Triton.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        return triton_selu(x)
