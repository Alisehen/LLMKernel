import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def post_process_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    scaling_factor,
    n_elements,
    C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Calculate channel index for bias broadcasting
    # For each element: idx = b*C*H*W + c*H*W + h*W + w
    spatial_size = H * W
    channel_size = C * spatial_size
    
    # Compute channel index (c)
    idx_in_batch = offsets % channel_size
    channel_idx = idx_in_batch // spatial_size
    
    # Load bias (broadcast across spatial dimensions)
    bias = tl.load(bias_ptr + channel_idx, mask=mask)
    
    # Fused operations
    # x = x + bias
    # x = clamp(x, 0, 1)
    # x = x * scaling_factor
    # x = clamp(x, 0, 1)
    # x = x / scaling_factor
    
    # Optimized version with common subexpression elimination
    x = x + bias
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    
    # Scale operations
    x = x * scaling_factor
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    x = x * (1.0 / scaling_factor)
    
    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)

def triton_post_process(
    x: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float
) -> torch.Tensor:
    # Ensure tensors are contiguous and on correct device
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert bias.is_contiguous(), "Bias tensor must be contiguous"
    
    # Reshape bias for broadcasting
    bias_reshaped = bias.view(-1)
    
    # Allocate output
    output = torch.empty_like(x)
    
    # Get tensor dimensions
    B, C, H, W = x.shape
    n_elements = output.numel()
    
    # Choose block size (power of 2)
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    post_process_kernel[grid](
        x, bias_reshaped, output,
        scaling_factor,
        n_elements,
        C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # First do the convolution (keep this in PyTorch)
        x = self.conv_transpose(x)
        
        # Use Triton kernel for post-processing
        return triton_post_process(x, self.bias, self.scaling_factor)
