import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def bias_add_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE_CHANNELS: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """
    Kernel for adding bias to conv+relu output.
    
    Grid dimensions:
    - pid_batch: batch index
    - pid_channel: starting channel index for block
    - pid_spatial: starting spatial index for block
    
    Each block processes a chunk of channels for a specific batch and spatial location.
    """
    pid_batch = tl.program_id(axis=0)
    pid_channel = tl.program_id(axis=1)
    pid_spatial = tl.program_id(axis=2)
    
    # Channel block indices
    channel_offsets = pid_channel * BLOCK_SIZE_CHANNELS + tl.arange(0, BLOCK_SIZE_CHANNELS)
    channel_mask = channel_offsets < channels
    
    # Spatial block indices
    spatial_offsets = pid_spatial * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_mask = spatial_offsets < spatial_size
    
    # Combine masks
    block_mask = channel_mask[:, None] & spatial_mask[None, :]
    
    # Base pointers for this batch
    batch_offset = pid_batch * channels * spatial_size
    base_input_ptr = input_ptr + batch_offset
    base_output_ptr = output_ptr + batch_offset
    
    # Load input chunk (channels x spatial)
    input_offsets = (channel_offsets[:, None] * spatial_size + spatial_offsets[None, :])
    input_chunk = tl.load(base_input_ptr + input_offsets, mask=block_mask, other=0.0)
    
    # Load bias (broadcast across spatial dimension)
    bias = tl.load(bias_ptr + channel_offsets, mask=channel_mask, other=0.0)
    
    # Add bias (broadcasting automatically)
    output_chunk = input_chunk + bias[:, None]
    
    # Store result
    tl.store(base_output_ptr + input_offsets, output_chunk, mask=block_mask)


def triton_bias_add(input_tensor: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for bias addition with broadcasting.
    
    Args:
        input_tensor: Shape [batch, channels, height, width]
        bias: Shape [channels, 1, 1]
    
    Returns:
        Tensor with same shape as input_tensor
    """
    # Ensure tensors are on GPU and contiguous
    input_tensor = input_tensor.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    batch_size, channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    # Reshape bias to 1D for kernel
    bias_reshaped = bias.view(channels)
    
    # Allocate output tensor
    output = torch.empty_like(input_tensor)
    
    # Block sizes (tuned for performance)
    BLOCK_SIZE_CHANNELS = 128
    BLOCK_SIZE_SPATIAL = 64
    
    # Grid configuration
    grid = (
        batch_size,  # batches
        triton.cdiv(channels, BLOCK_SIZE_CHANNELS),  # channel blocks
        triton.cdiv(spatial_size, BLOCK_SIZE_SPATIAL),  # spatial blocks
    )
    
    # Launch kernel
    bias_add_kernel[grid](
        input_tensor,
        bias_reshaped,
        output,
        batch_size,
        channels,
        spatial_size,
        BLOCK_SIZE_CHANNELS=BLOCK_SIZE_CHANNELS,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and adds a bias term.
    Triton version uses optimized bias addition kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        # Use Triton kernel for bias addition instead of PyTorch
        x = triton_bias_add(x, self.bias)
        return x
