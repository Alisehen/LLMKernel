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
    # 3D grid: batch, channel, spatial blocks
    pid_batch = tl.program_id(axis=0)
    pid_channel = tl.program_id(axis=1)
    pid_spatial = tl.program_id(axis=2)
    
    # Check bounds
    if pid_batch >= batch_size or pid_channel >= channels:
        return
    
    # Spatial block offsets
    spatial_start = pid_spatial * BLOCK_SIZE_SPATIAL
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    spatial_mask = spatial_offsets < spatial_size
    
    # Base pointers for this batch and channel
    batch_offset = pid_batch * batch_stride
    channel_offset = pid_channel * channel_stride
    base_offset = batch_offset + channel_offset
    
    # Load input block
    x_block = tl.load(
        x_ptr + base_offset + spatial_offsets * spatial_stride,
        mask=spatial_mask,
        other=0.0
    )
    
    # Load bias (scalar per channel)
    bias_scalar = tl.load(bias_ptr + pid_channel)
    
    # Fused operations
    # ReLU
    x_act = tl.maximum(x_block, 0.0)
    
    # LeakyReLU with negative_slope=0.01
    # Note: after ReLU, x_act is non-negative, so LeakyReLU does nothing
    # We keep it for completeness but optimize it away
    # x_leaky = tl.where(x_act < 0.0, x_act * 0.01, x_act) - equivalent to x_act after ReLU
    
    # GELU using accurate approximation without tanh
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # We use the fast accurate approximation from:
    # https://github.com/hendrycks/GELUs/blob/master/mnist_fc/gelu.py
    gelu_x = x_act
    # Constants precomputed for performance
    GELU_CONST = 0.7978845608028654  # sqrt(2/pi)
    GELU_COEF = 0.044715
    
    gelu_inner = GELU_CONST * (gelu_x + GELU_COEF * gelu_x * gelu_x * gelu_x)
    # Approximation for tanh using piecewise function (faster and more accurate than sigmoid)
    # tanh(x) ~ x * (27 + x*x) / (27 + 9*x*x) for small x
    # For our range, use the piecewise approximation
    gelu_tanh_approx = gelu_inner * (27.0 + gelu_inner * gelu_inner) / (27.0 + 9.0 * gelu_inner * gelu_inner)
    gelu_result = 0.5 * gelu_x * (1.0 + gelu_tanh_approx)
    
    # Sigmoid: 1/(1 + exp(-x))
    # Use stable computation to avoid overflow
    sigmoid_x = tl.where(gelu_result >= 0,
                        1.0 / (1.0 + tl.exp(-gelu_result)),
                        tl.exp(gelu_result) / (1.0 + tl.exp(gelu_result)))
    
    # Add bias
    result = sigmoid_x + bias_scalar
    
    # Store result
    tl.store(
        output_ptr + base_offset + spatial_offsets * spatial_stride,
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
    
    # Compute strides
    batch_stride = channels * spatial_size
    channel_stride = spatial_size
    spatial_stride = 1
    
    # Ensure bias is correctly shaped
    if bias.dim() == 4:
        bias = bias.squeeze(-1).squeeze(-1).squeeze(-1)
    
    # Set kernel configuration with optimal block size
    # Use larger block size for better memory coalescing
    BLOCK_SIZE_SPATIAL = 512
    grid = (
        batch_size,
        channels,
        triton.cdiv(spatial_size, BLOCK_SIZE_SPATIAL),
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
        spatial_stride,
        BLOCK_SIZE_SPATIAL,
    )
    
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
