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
    grid_channel: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CHANNEL: tl.constexpr,
    BLOCK_SIZE_HEIGHT: tl.constexpr,
    BLOCK_SIZE_WIDTH: tl.constexpr,
):
    # 3D grid: combine batch and channel dimensions
    pid_combined = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)
    
    # Decompose combined index
    pid_batch = pid_combined // grid_channel
    pid_channel = pid_combined % grid_channel
    
    # Create block ranges
    batch_range = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    channel_range = pid_channel * BLOCK_SIZE_CHANNEL + tl.arange(0, BLOCK_SIZE_CHANNEL)
    height_range = pid_h * BLOCK_SIZE_HEIGHT + tl.arange(0, BLOCK_SIZE_HEIGHT)
    width_range = pid_w * BLOCK_SIZE_WIDTH + tl.arange(0, BLOCK_SIZE_WIDTH)
    
    # Boundary masks
    batch_mask = batch_range < batch_size
    channel_mask = channel_range < out_channels
    height_mask = height_range < output_h
    width_mask = width_range < output_w
    
    # Initialize accumulator with proper broadcast pattern
    acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH), dtype=tl.float32)
    
    # Group configuration
    input_channels_per_group = in_channels // groups
    output_channels_per_group = out_channels // groups
    group_id = pid_channel // tl.cdiv(grid_channel, groups)
    
    # Main convolution loops
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Input positions with dilation
            input_h = height_range * stride_h + kh * dilation_h - padding_h
            input_w = width_range * stride_w + kw * dilation_w - padding_w
            
            # Input boundary checks
            input_h_mask = (input_h >= 0) & (input_h < height)
            input_w_mask = (input_w >= 0) & (input_w < width)
            
            # 2D spatial mask
            input_h_mask_2d = tl.broadcast_to(input_h_mask[:, None], (BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
            input_w_mask_2d = tl.broadcast_to(input_w_mask[None, :], (BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
            input_spatial_mask = input_h_mask_2d & input_w_mask_2d
            
            # Process each input channel in group
            channel_offset = group_id * input_channels_per_group
            
            for ic in range(input_channels_per_group):
                # Broadcast input indices
                input_h_broadcast = tl.broadcast_to(input_h[None, None, :, None], 
                                                   (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                input_w_broadcast = tl.broadcast_to(input_w[None, None, None, :], 
                                                   (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                
                # Input offsets with proper broadcasting
                batch_offset = tl.broadcast_to(batch_range[:, None, None, None] * input_batch_stride,
                                              (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                channel_offset_val = (channel_offset + ic) * input_channel_stride
                height_offset = input_h_broadcast * input_height_stride
                width_offset = input_w_broadcast * input_width_stride
                
                input_ptr_offset = batch_offset + channel_offset_val + height_offset + width_offset
                
                # Load input with broadcasted mask
                batch_mask_broadcast = tl.broadcast_to(batch_mask[:, None, None, None],
                                                      (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                spatial_mask_broadcast = tl.broadcast_to(input_spatial_mask[None, None, :, :],
                                                        (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                
                input_val = tl.load(
                    input_ptr + input_ptr_offset,
                    mask=batch_mask_broadcast & spatial_mask_broadcast,
                    other=0.0
                )
                
                # Weight offsets - ensure 1D vector for channels
                weight_channel_offset = channel_range * weight_output_channel_stride
                weight_input_offset = ic * weight_input_channel_stride
                weight_height_offset = kh * weight_height_stride
                weight_width_offset = kw * weight_width_stride
                
                weight_ptr_offset = weight_channel_offset + weight_input_offset + weight_height_offset + weight_width_offset
                
                # Load weight as 1D vector and broadcast
                weight_val_1d = tl.load(
                    weight_ptr + weight_ptr_offset,
                    mask=channel_mask,
                    other=0.0
                )
                weight_val = tl.broadcast_to(weight_val_1d[None, :, None, None],
                                            (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
                
                # Accumulate
                acc = acc + input_val * weight_val
    
    # Add bias with proper broadcasting
    if bias_ptr is not None:
        bias_val_1d = tl.load(
            bias_ptr + channel_range,
            mask=channel_mask,
            other=0.0
        )
        bias_val = tl.broadcast_to(bias_val_1d[None, :, None, None],
                                  (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
        acc = acc + bias_val
    
    # Apply division and leaky ReLU
    acc = acc / divisor
    acc = tl.where(acc >= 0, acc, acc * negative_slope)
    
    # Output offsets with broadcasting
    batch_output_offset = tl.broadcast_to(batch_range[:, None, None, None] * output_batch_stride,
                                         (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
    channel_output_offset = tl.broadcast_to(channel_range[None, :, None, None] * output_channel_stride,
                                           (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
    height_output_offset = tl.broadcast_to(height_range[None, None, :, None] * output_height_stride,
                                          (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
    width_output_offset = tl.broadcast_to(width_range[None, None, None, :] * output_width_stride,
                                         (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
    
    output_ptr_offset = batch_output_offset + channel_output_offset + height_output_offset + width_output_offset
    
    # Output mask
    output_mask = (
        tl.broadcast_to(batch_mask[:, None, None, None], (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH)) &
        tl.broadcast_to(channel_mask[None, :, None, None], (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH)) &
        tl.broadcast_to(height_mask[None, None, :, None], (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH)) &
        tl.broadcast_to(width_mask[None, None, None, :], (BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL, BLOCK_SIZE_HEIGHT, BLOCK_SIZE_WIDTH))
    )
    
    # Store result
    tl.store(
        output_ptr + output_ptr_offset,
        acc,
        mask=output_mask
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
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Output dimensions
    output_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    output_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1
    
    # Output tensor
    output = torch.empty(
        batch_size, out_channels, output_h, output_w,
        device=x.device, dtype=x.dtype
    )
    
    # Block sizes (optimized for V100/A100)
    BLOCK_SIZE_BATCH = min(batch_size, 1) if batch_size < 8 else min(batch_size, 2)
    BLOCK_SIZE_CHANNEL = 16 if out_channels % 16 == 0 else 8
    BLOCK_SIZE_HEIGHT = min(output_h, 8)
    BLOCK_SIZE_WIDTH = min(output_w, 8)
    
    # Grid dimensions
    grid_batch = triton.cdiv(batch_size, BLOCK_SIZE_BATCH)
    grid_channel = triton.cdiv(out_channels, BLOCK_SIZE_CHANNEL)
    grid_height = triton.cdiv(output_h, BLOCK_SIZE_HEIGHT)
    grid_width = triton.cdiv(output_w, BLOCK_SIZE_WIDTH)
    
    # Combine batch and channel
    grid_combined = grid_batch * grid_channel
    
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
    
    # Launch kernel
    conv_div_leaky_relu_kernel[(grid_combined, grid_height, grid_width)](
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
        grid_channel,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_CHANNEL=BLOCK_SIZE_CHANNEL,
        BLOCK_SIZE_HEIGHT=BLOCK_SIZE_HEIGHT,
        BLOCK_SIZE_WIDTH=BLOCK_SIZE_WIDTH,
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
