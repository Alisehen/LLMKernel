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
    VEC_SIZE: tl.constexpr,
):
    # 3D grid: batch, channel, spatial blocks
    pid_batch = tl.program_id(axis=0)
    pid_channel = tl.program_id(axis=1)
    pid_spatial = tl.program_id(axis=2)
    
    # Check bounds
    if pid_batch >= batch_size or pid_channel >= channels:
        return
    
    # Spatial block in elements
    spatial_start = pid_spatial * BLOCK_SIZE_SPATIAL
    # Convert to vector index
    vec_start = spatial_start // VEC_SIZE
    BLOCK_SIZE_VEC = BLOCK_SIZE_SPATIAL // VEC_SIZE
    vec_index = vec_start + tl.arange(0, BLOCK_SIZE_VEC)
    vec_mask = vec_index < (spatial_size + VEC_SIZE - 1) // VEC_SIZE
    
    # Base pointers for this batch and channel
    batch_offset = pid_batch * batch_stride
    channel_offset = pid_channel * channel_stride
    base_offset = batch_offset + channel_offset
    
    # Load bias (scalar per channel)
    bias_scalar = tl.load(bias_ptr + pid_channel)
    
    # Element indices for vector load
    element_index = vec_index * VEC_SIZE
    # Create 2D offsets for vector elements (BLOCK_SIZE_VEC, VEC_SIZE)
    vec_offsets = element_index[:, None] + tl.arange(0, VEC_SIZE)[None, :]
    vec_offsets_mask = vec_offsets < spatial_size
    
    # Load input vectors
    base_ptr = x_ptr + base_offset
    x_vecs = tl.load(
        base_ptr + vec_offsets * spatial_stride,
        mask=vec_offsets_mask,
        other=0.0
    )
    
    # Fused operations applied to each element in the vector
    # ReLU
    x_act = tl.maximum(x_vecs, 0.0)
    
    # GELU using fast accurate approximation
    GELU_CONST = 0.7978845608028654  # sqrt(2/pi)
    GELU_COEF = 0.044715
    
    gelu_x = x_act
    gelu_inner = GELU_CONST * (gelu_x + GELU_COEF * gelu_x * gelu_x * gelu_x)
    # tanh approximation optimized for vector operations
    gelu_tanh_approx = gelu_inner * (27.0 + gelu_inner * gelu_inner) / (27.0 + 9.0 * gelu_inner * gelu_inner)
    gelu_result = 0.5 * gelu_x * (1.0 + gelu_tanh_approx)
    
    # Sigmoid with stable computation
    sigmoid_result = tl.where(
        gelu_result >= 0,
        1.0 / (1.0 + tl.exp(-gelu_result)),
        tl.exp(gelu_result) / (1.0 + tl.exp(gelu_result))
    )
    
    # Add bias (broadcast scalar to vector)
    result = sigmoid_result + bias_scalar
    
    # Store result vectors
    tl.store(
        base_ptr + vec_offsets * spatial_stride,
        result,
        mask=vec_offsets_mask
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
    
    # Vectorized configuration
    VEC_SIZE = 4
    BLOCK_SIZE_SPATIAL = 512  # elements per block
    BLOCK_SIZE_VEC = BLOCK_SIZE_SPATIAL // VEC_SIZE  # vectors per block
    
    grid = (
        batch_size,
        channels,
        triton.cdiv(spatial_size, BLOCK_SIZE_SPATIAL),
    )
    
    # Autotune configurations around optimal points
    # Configs: (num_warps, num_stages)
    configs = [
        {'num_warps': 4, 'num_stages': 2},  # Default vectorized
        {'num_warps': 4, 'num_stages': 3},  # Current
        {'num_warps': 4, 'num_stages': 4},  # +1 stage
        {'num_warps': 2, 'num_stages': 3},  # Fewer warps
    ]
    
    # Use autotune to select best configuration
    best_config = None
    best_time = float('inf')
    
    for config in configs:
        try:
            # Warmup
            fused_activation_bias_kernel[grid](
                x_reshaped, bias, output_reshaped,
                batch_size, channels, spatial_size,
                batch_stride, channel_stride, spatial_stride,
                BLOCK_SIZE_SPATIAL, VEC_SIZE,
                num_warps=config['num_warps'],
                num_stages=config['num_stages']
            )
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                fused_activation_bias_kernel[grid](
                    x_reshaped, bias, output_reshaped,
                    batch_size, channels, spatial_size,
                    batch_stride, channel_stride, spatial_stride,
                    BLOCK_SIZE_SPATIAL, VEC_SIZE,
                    num_warps=config['num_warps'],
                    num_stages=config['num_stages']
                )
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 10
            
            if elapsed < best_time:
                best_time = elapsed
                best_config = config
        except Exception:
            continue
    
    # Use best configuration (fallback to default if none found)
    if best_config is None:
        best_config = {'num_warps': 4, 'num_stages': 3}
    
    # Launch with best configuration
    fused_activation_bias_kernel[grid](
        x_reshaped, bias, output_reshaped,
        batch_size, channels, spatial_size,
        batch_stride, channel_stride, spatial_stride,
        BLOCK_SIZE_SPATIAL, VEC_SIZE,
        num_warps=best_config['num_warps'],
        num_stages=best_config['num_stages']
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
