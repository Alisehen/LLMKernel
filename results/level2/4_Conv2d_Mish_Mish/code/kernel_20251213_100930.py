import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def double_mish_kernel_optimized(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr,
):
    """
    Optimized double Mish kernel with:
    - Vectorized loads/stores (4 elements per thread)
    - Optimized math using (1+exp(x))² identity
    - Increased warps for better latency hiding
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE * VECTOR_WIDTH
    offsets = start + tl.arange(0, BLOCK_SIZE * VECTOR_WIDTH)
    mask = offsets < n_elements
    
    # Vectorized load (4 elements per thread)
    x_vec = tl.load(input_ptr + offsets, mask=mask)
    
    # First Mish: x * tanh(softplus(x))
    # Optimized using: exp(2*softplus(x)) = (1 + exp(x))²
    t1 = tl.exp(x_vec)
    s1 = 1.0 + t1
    exp_2s1 = s1 * s1  # (1 + exp(x))²
    tanh_s1 = (exp_2s1 - 1.0) / (exp_2s1 + 1.0)
    y_vec = x_vec * tanh_s1
    
    # Second Mish: same optimization
    t2 = tl.exp(y_vec)
    s2 = 1.0 + t2
    exp_2s2 = s2 * s2  # (1 + exp(y))²
    tanh_s2 = (exp_2s2 - 1.0) / (exp_2s2 + 1.0)
    z_vec = y_vec * tanh_s2
    
    # Vectorized store
    tl.store(output_ptr + offsets, z_vec, mask=mask)

def triton_double_mish_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper for double Mish activation.
    Uses vectorized kernel with tuned parameters.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Tuned parameters for Ada Lovelace (4090)
    # - Vector width 4: Maximizes memory throughput
    # - Block size 512: Better occupancy (16 warps per block)
    # - num_warps 16: 16 warps = 512 threads (optimal for Ada)
    # - num_stages 3: Prefetch more data to hide memory latency
    BLOCK_SIZE = 512
    VECTOR_WIDTH = 4
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VECTOR_WIDTH']),)
    
    # Launch with optimized compilation options
    double_mish_kernel_optimized[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        VECTOR_WIDTH=VECTOR_WIDTH,
        num_warps=16,
        num_stages=3
    )
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    Uses fused Triton kernels for the Mish activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Use optimized fused double mish kernel
        x = triton_double_mish_optimized(x)
        return x
