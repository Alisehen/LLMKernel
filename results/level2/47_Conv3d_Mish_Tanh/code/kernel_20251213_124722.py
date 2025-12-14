import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'VECTORIZE': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'VECTORIZE': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTORIZE': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'VECTORIZE': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256, 'VECTORIZE': 4}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def mish_tanh_kernel_3d_optimized(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr,
):
    """
    Optimized Mish+Tanh kernel with vectorization.
    """
    pid = tl.program_id(axis=0)
    
    # Vectorized processing
    for vec_idx in tl.range(VECTORIZE):
        vec_offset = pid * BLOCK_SIZE * VECTORIZE + vec_idx * BLOCK_SIZE
        offsets = vec_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # Mish activation: x * tanh(softplus(x))
        # softplus(x) = ln(1 + exp(x))
        # Optimized computation using log1p alternative
        exp_neg_x = tl.exp(-tl.where(x > 0, x, -x))
        
        # Compute softplus(x) = ln(1 + exp(x))
        softplus_x = tl.where(
            x > 0,
            x + tl.log(1.0 + exp_neg_x),  # x + ln(1 + exp(-x)) for x > 0
            tl.log(1.0 + tl.exp(x))       # ln(1 + exp(x)) for x <= 0
        )
        
        # tanh(softplus(x))
        tanh_softplus = tl.math.tanh(softplus_x)
        mish_x = x * tanh_softplus
        
        # Apply tanh to result
        result = tl.math.tanh(mish_x)
        
        # Store result
        tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def mish_tanh_kernel_3d_large(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for large tensors with increased register usage.
    """
    pid = tl.program_id(axis=0)
    
    # Precompute offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute with minimal branches
    # Alternative computation for softplus
    exp_x = tl.exp(x)
    softplus_x = tl.where(
        x > 0,
        x + tl.log(1.0 + tl.exp(-x)),
        tl.log(1.0 + exp_x)
    )
    
    # Fast tanh using single exp
    exp_2x = tl.exp(2.0 * softplus_x)
    tanh_softplus = (exp_2x - 1.0) / (exp_2x + 1.0)
    mish_x = x * tanh_softplus
    
    # Final tanh
    exp_2mish = tl.exp(2.0 * mish_x)
    result = (exp_2mish - 1.0) / (exp_2mish + 1.0)
    
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_mish_tanh_3d_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper with automatic kernel selection.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Heuristic for kernel selection
    if n_elements > 10_000_000:
        # Use large kernel for very large tensors
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        mish_tanh_kernel_3d_large[grid](
            x, output, n_elements, 
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2
        )
    else:
        # Use autotuned kernel for smaller tensors
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta.get('VECTORIZE', 1)),)
        mish_tanh_kernel_3d_optimized[grid](x, output, n_elements)
    
    return output


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    Uses optimized Triton kernels for the activation operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = triton_mish_tanh_3d_optimized(x)
        return x
