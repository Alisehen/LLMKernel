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
    # Use log1p for numerical stability
    softplus_x = tl.log1p(tl.exp(x))
    tanh_softplus = tl.tanh(softplus_x)
    mish_x = x * tanh_softplus
    
    # Tanh activation
    result = tl.tanh(mish_x)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


def triton_mish_tanh_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Mish+Tanh kernel.
    """
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose block size as a power of 2
    BLOCK_SIZE = 1024
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
