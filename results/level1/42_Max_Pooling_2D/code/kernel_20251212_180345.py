import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool2d_kernel(
    x_ptr,
    output_ptr,
    # Tensor dimensions
    batch_stride_x, channel_stride_x, height_stride_x, width_stride_x,
    batch_stride_out, channel_stride_out, height_stride_out, width_stride_out,
    batch_size, channels, height, width,
    pooled_height, pooled_width,
    # Pooling parameters
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    # Grid parameters
    grid_ph: tl.constexpr,
    grid_pw: tl.constexpr,
    # Block parameters
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize across output positions and channels
    pid_batch = tl.program_id(0)
    pid_channel = tl.program_id(1)
    pid_combined = tl.program_id(2)
    
    # Split combined dimension into ph and pw blocks
    pid_ph = pid_combined // grid_pw
    pid_pw = pid_combined % grid_pw
    
    # Check boundaries
    if pid_batch >= batch_size:
        return
    
    # Create block of output positions in H and W dimensions
    ph_block_start = pid_ph * BLOCK_SIZE_H
    ph_offsets = ph_block_start + tl.arange(0, BLOCK_SIZE_H)
    ph_mask = ph_offsets < pooled_height
    
    pw_block_start = pid_pw * BLOCK_SIZE_W
    pw_offsets = pw_block_start + tl.arange(0, BLOCK_SIZE_W)
    pw_mask = pw_offsets < pooled_width
    
    # Process multiple channels per block
    channel_block_start = pid_channel * BLOCK_SIZE_C
    channel_offsets = channel_block_start + tl.arange(0, BLOCK_SIZE_C)
    channel_mask = channel_offsets < channels
    
    # Initialize output with -inf
    output_block = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), 
                          float('-inf'), dtype=tl.float32)
    
    # For each position in the pooling window
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Compute input positions with dilation
            h_idx = (ph_offsets[:, None, None] * stride_h - pad_h + 
                    kh * dilation_h)
            w_idx = (pw_offsets[None, :, None] * stride_w - pad_w + 
                    kw * dilation_w)
            
            # Create mask for valid input positions
            h_valid = (h_idx >= 0) & (h_idx < height)
            w_valid = (w_idx >= 0) & (w_idx < width)
            valid_mask = h_valid & w_valid  # [BLOCK_SIZE_H, BLOCK_SIZE_W, 1]
            
            # Compute input pointers
            x_ptrs = (
                x_ptr +
                pid_batch * batch_stride_x +
                channel_offsets[None, None, :] * channel_stride_x +
                h_idx * height_stride_x +
                w_idx * width_stride_x
            )
            
            # Load input values with boundary checking
            # Create proper mask shape [BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C]
            expanded_valid_mask = tl.broadcast_to(valid_mask, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
            expanded_channel_mask = tl.broadcast_to(channel_mask[None, None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
            load_mask = expanded_valid_mask & expanded_channel_mask
            
            x_vals = tl.load(x_ptrs, mask=load_mask, other=float('-inf'))
            
            # Update max (invalid positions are -inf, so they don't affect the max)
            output_block = tl.maximum(output_block, x_vals)
    
    # Store results
    # Create output pointer offsets
    ph_expanded = tl.broadcast_to(ph_offsets[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    pw_expanded = tl.broadcast_to(pw_offsets[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    ch_expanded = tl.broadcast_to(channel_offsets[None, None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    
    output_ptrs = (
        output_ptr +
        pid_batch * batch_stride_out +
        ch_expanded * channel_stride_out +
        ph_expanded * height_stride_out +
        pw_expanded * width_stride_out
    )
    
    # Create output mask with proper broadcasting
    output_mask = (
        tl.broadcast_to(ph_mask[:, None, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) &
        tl.broadcast_to(pw_mask[None, :, None], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C)) &
        tl.broadcast_to(channel_mask[None, None, :], (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    )
    
    tl.store(output_ptrs, output_block, mask=output_mask)


def triton_max_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Triton implementation of 2D max pooling."""
    if stride is None:
        stride = kernel_size
    
    # Ensure kernel_size is tuple
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    # Ensure other parameters are tuples
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Compute output dimensions
    batch_size, channels, height, width = x.shape
    pooled_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    pooled_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(
        (batch_size, channels, pooled_height, pooled_width),
        device=x.device,
        dtype=x.dtype
    )
    
    # Strides for tensor access
    batch_stride_x = x.stride(0)
    channel_stride_x = x.stride(1)
    height_stride_x = x.stride(2)
    width_stride_x = x.stride(3)
    
    batch_stride_out = output.stride(0)
    channel_stride_out = output.stride(1)
    height_stride_out = output.stride(2)
    width_stride_out = output.stride(3)
    
    # Configure kernel launch
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 4
    
    grid_ph = triton.cdiv(pooled_height, BLOCK_SIZE_H)
    grid_pw = triton.cdiv(pooled_width, BLOCK_SIZE_W)
    grid_combined = grid_ph * grid_pw
    
    grid = (
        batch_size,
        triton.cdiv(channels, BLOCK_SIZE_C),
        grid_combined,
    )
    
    # Launch kernel
    max_pool2d_kernel[grid](
        x,
        output,
        batch_stride_x, channel_stride_x, height_stride_x, width_stride_x,
        batch_stride_out, channel_stride_out, height_stride_out, width_stride_out,
        batch_size, channels, height, width,
        pooled_height, pooled_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        grid_ph=grid_ph,
        grid_pw=grid_pw,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Max Pooling 2D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        return triton_max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
