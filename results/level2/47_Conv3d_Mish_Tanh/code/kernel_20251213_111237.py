import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mish_tanh_kernel_3d(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that applies Mish activation followed by Tanh activation.
    Mish(x) = x * tanh(softplus(x)), where softplus(x) = log(1 + exp(x))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Mish activation: x * tanh(softplus(x))
    # Compute softplus(x) = log(1 + exp(x)) using stable implementation
    # For numerical stability, handle both positive and negative cases
    # Use math formulas instead of unavailable tl.log1p
    softplus_x = tl.where(
        x > 0,
        x + tl.log(1.0 + tl.exp(-x)),  # Stable for positive x
        tl.log(1.0 + tl.exp(x))        # Direct computation for negative x
    )
    
    # Implement tanh using exp formulas since tl.tanh is not available
    # tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    tanh_softplus = tl.where(
        softplus_x > 0,
        # For positive values: use formula with negative exponent for stability
        1.0 - 2.0 / (1.0 + tl.exp(2.0 * softplus_x)),
        # For negative values: use standard formula
        (tl.exp(2.0 * softplus_x) - 1.0) / (tl.exp(2.0 * softplus_x) + 1.0)
    )
    mish_x = x * tanh_softplus
    
    # Tanh activation for the result
    # Apply tanh similarly to mish_x
    result = tl.where(
        mish_x > 0,
        1.0 - 2.0 / (1.0 + tl.exp(2.0 * mish_x)),
        (tl.exp(2.0 * mish_x) - 1.0) / (tl.exp(2.0 * mish_x) + 1.0)
    )
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_mish_tanh_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Mish+Tanh kernel.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose optimal block size based on hardware
    # Using power of 2 up to 1024, adjust based on tensor size
    BLOCK_SIZE = 1024
    while BLOCK_SIZE > n_elements and BLOCK_SIZE > 32:
        BLOCK_SIZE //= 2
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mish_tanh_kernel_3d[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    Uses Triton kernel for the activation operations.
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
