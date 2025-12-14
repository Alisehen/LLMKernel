import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)

    # Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    x = x * sigmoid_x
    
    # Divide by 2.0
    x = x * 0.5
    
    # Clamp between -1 and 1
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Tanh activation using exp function
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * x)
    x = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Clamp between -1 and 1
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)

    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized block size for compute-bound operations
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        # Fused activation using Triton kernel
        x = triton_fused_activation(x)
        return x
