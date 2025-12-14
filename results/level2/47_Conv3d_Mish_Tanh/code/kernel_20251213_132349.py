import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def mish_tanh_kernel_3d(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that applies Mish activation followed by Tanh activation.
    Uses autotuning for num_warps and num_stages based on NCU metrics.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with vectorized memory access
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Mish activation: x * tanh(softplus(x))
    # Optimized: Use tl.exp for better numerical stability and performance
    # Handle positive and negative cases with max/min operations
    exp_neg_x = tl.exp(-tl.where(x > 0, x, -x))  # exp(-|x|)
    
    # Compute softplus(x) = log(1 + exp(x))
    # For x >= 0: x + log(1 + exp(-x))
    # For x < 0: log(1 + exp(x))
    softplus_x = tl.where(
        x >= 0,
        x + tl.log(1.0 + exp_neg_x),
        tl.log(1.0 + tl.exp(x))
    )
    
    # Compute tanh(softplus_x) using optimized formula
    # tanh(x) = 1 - 2/(1 + exp(2x)) for x >= 0
    #         = (exp(2x) - 1)/(exp(2x) + 1) for x < 0
    exp_2x = tl.exp(2.0 * softplus_x)
    tanh_softplus = tl.where(
        softplus_x >= 0,
        1.0 - 2.0 / (1.0 + exp_2x),
        (exp_2x - 1.0) / (exp_2x + 1.0)
    )
    
    mish_x = x * tanh_softplus
    
    # Tanh activation for the result
    # Reuse computation pattern for efficiency
    exp_2mish = tl.exp(2.0 * mish_x)
    result = tl.where(
        mish_x >= 0,
        1.0 - 2.0 / (1.0 + exp_2mish),
        (exp_2mish - 1.0) / (exp_2mish + 1.0)
    )
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_mish_tanh_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the optimized Mish+Tanh kernel with autotuning.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Fixed BLOCK_SIZE=1024 for optimal occupancy on Ada Lovelace
    # Autotuning will handle num_warps and num_stages variations
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mish_tanh_kernel_3d[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    Uses optimized Triton kernel with autotuning for the activation operations.
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
        x = triton_mish_tanh_3d(x)
        return x
