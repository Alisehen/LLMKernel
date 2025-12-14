import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_div_leaky_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    divisor,
    negative_slope,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    batch_size, in_channels, out_channels,
    height, width,
    kernel_h, kernel_w,
    output_h, output_w,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    weight_output_channel_stride, weight_input_channel_stride, weight_height_stride, weight_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CHANNEL: tl.constexpr,
    BLOCK_SIZE_HEIGHT: tl.constexpr,
    BLOCK_SIZE_WIDTH: tl.constexpr,
    USE_BIAS: tl.constexpr,
    GROUP_IN_CHANNELS: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Optimized convolution with division and leaky ReLU activation."""
    # 3D grid: (batch, channel, height*width)
    pid_batch = tl.program_id(axis=0)
    pid_channel = tl.program_id(axis=1)
    pid_spatial = tl.program_id(axis=2)
    
    # Calculate spatial block
    blocks_per_row = tl.cdiv(output_w, BLOCK_SIZE_WIDTH)
    pid_h = pid_spatial // blocks_per_row
    pid_w = pid_spatial % blocks_per_row
    
    # Create block ranges with efficient indexing
    batch_range = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    channel_range = pid_channel * BLOCK_SIZE_CHANNEL + tl.arange(0, BLOCK_SIZE_CHANNEL)
    height_range = pid_h * BLOCK_SIZE_HEIGHT + tl.arange(0, BLOCK_SIZE_HEIGHT)
    width_range = pid_w * BLOCK_SIZE_WIDTH + tl.arange(0, BLOCK_SIZE_WIDTH)
    
    # Boundary masks
    batch_mask = batch_range < batch_size
    channel_mask = channel_range < out_channels
    height_mask = height_range < output_h
    width_mask = width_range < output_w
    
    # Group configuration
    output_channels_per_group = out_channels // groups
    channel_blocks_per_group = tl.cdiv(output_channels_per_group, BLOCK_SIZE_CHANNEL)
    group_id = pid_channel // channel_blocks_per_group
    group_channel_offset = group_id * output_channels_per_group
    
    # Initialize accumulator with proper type
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH), 
                   dtype=tl.float32)
    
    # Precompute input channel start for this group
    input_channel_start = group_id * GROUP_IN_CHANNELS
    
    # Loop over input channels with pipelining for better memory latency hiding
    for ic in range(GROUP_IN_CHANNELS):
        input_channel_idx = input_channel_start + ic
        
        # Process kernel spatial dimensions with loop unrolling hint
        for kh_idx in range(kernel_h):
            kh = kh_idx * dilation_h
            for kw_idx in range(kernel_w):
                kw = kw_idx * dilation_w
                
                # Input positions with boundary checks
                input_h = height_range * stride_h + kh - padding_h
                input_w = width_range * stride_w + kw - padding_w
                
                input_h_mask = (input_h >= 0) & (input_h < height)
                input_w_mask = (input_w >= 0) & (input_w < width)
                spatial_mask = input_h_mask[:, None] & input_w_mask[None, :]
                
                # Load input values with coalesced access pattern
                input_offsets = (
                    batch_range[:, None, None, None] * input_batch_stride +
                    input_channel_idx * input_channel_stride +
                    input_h[None, None, :, None] * input_height_stride +
                    input_w[None, None, None, :] * input_width_stride
                )
                
                input_val = tl.load(
                    input_ptr + input_offsets,
                    mask=batch_mask[:, None, None, None] & spatial_mask[None, None, :, :],
                    other=0.0,
                    cache_modifier=".cg"  # Cache global for better L1 hit rate
                )
                
                # Load weight values with proper masking
                weight_channel_range = channel_range + group_channel_offset
                weight_offsets = (
                    weight_channel_range[None, :, None, None] * weight_output_channel_stride +
                    ic * weight_input_channel_stride +
                    kh_idx * weight_height_stride +
                    kw_idx * weight_width_stride
                )
                
                weight_val = tl.load(
                    weight_ptr + weight_offsets,
                    mask=channel_mask[None, :, None, None],
                    other=0.0,
                    cache_modifier=".ca"  # Cache all for weight reuse
                )
                
                # FMA operation
                acc = tl.math.fma(input_val, weight_val, acc)
    
    # Add bias if present
    if USE_BIAS:
        bias_val = tl.load(
            bias_ptr + channel_range + group_channel_offset,
            mask=channel_mask,
            other=0.0,
            cache_modifier=".ca"
        )
        acc = acc + bias_val[None, :, None, None]
    
    # Apply division and leaky ReLU
    acc = acc / divisor
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Compute output offsets
    output_offsets = (
        batch_range[:, None, None, None] * output_batch_stride +
        (channel_range + group_channel_offset)[None, :, None, None] * output_channel_stride +
        height_range[None, None, :, None] * output_height_stride +
        width_range[None, None, None, :] * output_width_stride
    )
    
    # Store result with coalesced writes
    tl.store(
        output_ptr + output_offsets,
        acc,
        mask=(
            batch_mask[:, None, None, None] &
            channel_mask[None, :, None, None] &
            height_mask[None, None, :, None] &
            width_mask[None, None, None, :]
        ),
        cache_modifier=".cg"
    )


def triton_conv_div_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    divisor: float,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Optimized wrapper for convolution with division and leaky ReLU."""
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, weight_in_channels, kernel_h, kernel_w = weight.shape
    
    # Output dimensions
    output_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    output_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Output tensor
    output = torch.empty(
        batch_size, out_channels, output_h, output_w,
        device=x.device, dtype=x.dtype
    )
    
    # Optimized block sizes for Ada Lovelace (RTX 4090)
    # Based on NCU metrics: increased parallelism for better SM utilization
    BLOCK_SIZE_BATCH = 1  # Reduced for more parallelism
    BLOCK_SIZE_CHANNEL = 32  # Kept for channel parallelism
    BLOCK_SIZE_HEIGHT = 4  # Reduced to increase spatial blocks
    BLOCK_SIZE_WIDTH = 4   # Reduced to increase spatial blocks
    
    # Ensure valid block sizes
    BLOCK_SIZE_BATCH = min(BLOCK_SIZE_BATCH, batch_size)
    BLOCK_SIZE_CHANNEL = min(BLOCK_SIZE_CHANNEL, out_channels)
    BLOCK_SIZE_HEIGHT = min(BLOCK_SIZE_HEIGHT, output_h)
    BLOCK_SIZE_WIDTH = min(BLOCK_SIZE_WIDTH, output_w)
    
    # Grid dimensions - 3D for maximum SM utilization
    grid_batch = triton.cdiv(batch_size, BLOCK_SIZE_BATCH)
    grid_channel = triton.cdiv(out_channels, BLOCK_SIZE_CHANNEL)
    grid_spatial = triton.cdiv(output_h, BLOCK_SIZE_HEIGHT) * triton.cdiv(output_w, BLOCK_SIZE_WIDTH)
    
    # Ensure we have enough parallelism for 128 SMs
    total_blocks = grid_batch * grid_channel * grid_spatial
    target_blocks = 512  # Minimum for good SM utilization
    
    if total_blocks < target_blocks:
        # Reduce block sizes to increase parallelism
        if grid_batch * grid_channel < 128:
            BLOCK_SIZE_HEIGHT = max(1, BLOCK_SIZE_HEIGHT // 2)
            BLOCK_SIZE_WIDTH = max(1, BLOCK_SIZE_WIDTH // 2)
            if BLOCK_SIZE_HEIGHT * BLOCK_SIZE_WIDTH == 1:
                BLOCK_SIZE_CHANNEL = max(8, BLOCK_SIZE_CHANNEL // 2)
        
        # Recompute grid
        BLOCK_SIZE_HEIGHT = min(BLOCK_SIZE_HEIGHT, output_h)
        BLOCK_SIZE_WIDTH = min(BLOCK_SIZE_WIDTH, output_w)
        BLOCK_SIZE_CHANNEL = min(BLOCK_SIZE_CHANNEL, out_channels)
        
        grid_batch = triton.cdiv(batch_size, BLOCK_SIZE_BATCH)
        grid_channel = triton.cdiv(out_channels, BLOCK_SIZE_CHANNEL)
        grid_spatial = triton.cdiv(output_h, BLOCK_SIZE_HEIGHT) * triton.cdiv(output_w, BLOCK_SIZE_WIDTH)
    
    # Strides
    input_batch_stride = x.stride(0)
    input_channel_stride = x.stride(1)
    input_height_stride = x.stride(2)
    input_width_stride = x.stride(3)
    
    weight_output_channel_stride = weight.stride(0)
    weight_input_channel_stride = weight.stride(1)
    weight_height_stride = weight.stride(2)
    weight_width_stride = weight.stride(3)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)
    
    # Group configuration
    input_channels_per_group = in_channels // groups
    
    # Launch kernel with optimized configuration
    conv_div_leaky_relu_kernel[(grid_batch, grid_channel, grid_spatial)](
        x,
        weight,
        bias,
        output,
        divisor,
        0.01,  # negative_slope
        stride, stride,
        padding, padding,
        dilation, dilation,
        groups,
        batch_size, in_channels, out_channels,
        height, width,
        kernel_h, kernel_w,
        output_h, output_w,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
        weight_output_channel_stride, weight_input_channel_stride, weight_height_stride, weight_width_stride,
        output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_CHANNEL=BLOCK_SIZE_CHANNEL,
        BLOCK_SIZE_HEIGHT=BLOCK_SIZE_HEIGHT,
        BLOCK_SIZE_WIDTH=BLOCK_SIZE_WIDTH,
        USE_BIAS=bias is not None,
        GROUP_IN_CHANNELS=input_channels_per_group,
        num_warps=8,  # Optimized for 256 threads per block (8 warps)
        num_stages=3,  # Increased for better memory latency hiding
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        return triton_conv_div_leaky_relu(
            x,
            self.conv.weight,
            self.conv.bias,
            self.divisor,
            stride=self.conv.stride[0],
            padding=self.conv.padding[0],
            dilation=self.conv.dilation[0],
            groups=self.conv.groups,
        )
