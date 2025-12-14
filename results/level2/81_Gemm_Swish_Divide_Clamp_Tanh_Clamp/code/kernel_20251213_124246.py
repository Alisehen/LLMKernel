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

    # Load input with vectorized access
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Swish: x * sigmoid(x)
    # Use fast approximation: sigmoid(x) ≈ 0.5 * tanh(0.5 * x) + 0.5
    half_x = x * 0.5
    tanh_half = tl.tanh(half_x)
    sigmoid_x = tanh_half * 0.5 + 0.5
    x = x * sigmoid_x
    
    # Divide by 2.0
    x = x * 0.5
    
    # Clamp between -1 and 1
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    
    # Tanh activation using hardware tanh
    x = tl.tanh(x)
    
    # Clamp between -1 and 1 (redundant but kept for correctness)
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)

    # Store result with vectorized store
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Optimized block size for memory efficiency
    # Reduced to improve occupancy and hide memory latency
    BLOCK_SIZE = 512
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Tuned parameters based on NCU metrics
    # - num_warps=4 reduces register pressure
    # - num_stages=3 improves memory latency hiding
    fused_activation_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3
    )
    
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
        # Fused activation using optimized Triton kernel
        x = triton_fused_activation(x)
        return x
