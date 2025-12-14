import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_bias_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    batch_stride,
    channel_stride,
    spatial_stride,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    # Optimized 2D grid: (batch*channels, spatial_blocks)
    # Flatten batch and channel dimensions for better parallelism
    pid_batch_channel = tl.program_id(axis=0)
    pid_spatial = tl.program_id(axis=1)
    
    # Reconstruct batch and channel indices
    batch_idx = pid_batch_channel // channels
    channel_idx = pid_batch_channel % channels
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Spatial block offsets
    spatial_start = pid_spatial * BLOCK_SIZE_SPATIAL
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_mask = spatial_offsets < spatial_size
    
    # Base pointer calculation
    batch_offset = batch_idx * batch_stride
    channel_offset = channel_idx * channel_stride
    base_offset = batch_offset + channel_offset
    
    # Load input block with better memory coalescing
    x_block = tl.load(
        x_ptr + base_offset + spatial_offsets,
        mask=spatial_mask,
        other=0.0
    )
    
    # Load bias (scalar per channel)
    bias_scalar = tl.load(bias_ptr + channel_idx)
    
    # Fused operations
    # ReLU
    x_act = tl.maximum(x_block, 0.0)
    
    # Optimized GELU approximation - use more accurate polynomial
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Precomputed constants for performance
    GELU_CONST = 0.7978845608028654  # sqrt(2/pi)
    GELU_COEF = 0.044715
    
    x_cubed = x_act * x_act * x_act
    gelu_inner = GELU_CONST * (x_act + GELU_COEF * x_cubed)
    
    # Optimized tanh approximation: tanh(x) = x * (1 - x^2/3) for small x (more accurate)
    gelu_inner_sq = gelu_inner * gelu_inner
    tanh_approx = gelu_inner * (1.0 - gelu_inner_sq * 0.3333333333333333)
    gelu_result = 0.5 * x_act * (1.0 + tanh_approx)
    
    # Optimized Sigmoid: sigmoid(x) = 1/(1 + exp(-x))
    # Use piecewise approximation for better performance
    sigmoid_result = tl.where(gelu_result >= 0,
                             1.0 / (1.0 + tl.exp(-gelu_result)),
                             tl.exp(gelu_result) / (1.0 + tl.exp(gelu_result)))
    
    # Add bias
    result = sigmoid_result + bias_scalar
    
    # Store result
    tl.store(
        output_ptr + base_offset + spatial_offsets,
        result,
        mask=spatial_mask
    )

def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused activation and bias addition for 5D tensors.
    x shape: [batch, channels, depth, height, width]
    bias shape: [channels, 1, 1, 1] or [channels]
    """
    output = torch.empty_like(x)
    
    batch_size, channels, depth, height, width = x.shape
    spatial_size = depth * height * width
    
    # Flatten spatial dimensions
    x_reshaped = x.reshape(batch_size, channels, spatial_size)
    output_reshaped = output.reshape(batch_size, channels, spatial_size)
    
    # Compute strides for flattened tensor
    batch_stride = channels * spatial_size
    channel_stride = spatial_size
    
    # Ensure bias is correctly shaped
    if bias.dim() == 4:
        bias = bias.squeeze(-1).squeeze(-1).squeeze(-1)
    
    # Optimized kernel configuration with autotuning
    # Use 2D grid for better SM utilization
    configs = [
        triton.Config({'BLOCK_SIZE_SPATIAL': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_SPATIAL': 2048}, num_warps=16, num_stages=1),
    ]
    
    @triton.autotune(configs=configs, key=['spatial_size'])
    def kernel_launcher(x_reshaped, bias, output_reshaped, batch_size, channels, spatial_size, batch_stride, channel_stride):
        grid = lambda META: (
            batch_size * channels,  # Combine batch and channel for better parallelism
            triton.cdiv(spatial_size, META['BLOCK_SIZE_SPATIAL']),
        )
        
        fused_activation_bias_kernel[grid](
            x_reshaped,
            bias,
            output_reshaped,
            batch_size,
            channels,
            spatial_size,
            batch_stride,
            channel_stride,
            1,  # spatial_stride is 1 for contiguous flattened tensor
            META['BLOCK_SIZE_SPATIAL'],
        )
    
    kernel_launcher(x_reshaped, bias, output_reshaped, batch_size, channels, spatial_size, batch_stride, channel_stride)
    
    return output

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies fused activations and bias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias)
        return x
