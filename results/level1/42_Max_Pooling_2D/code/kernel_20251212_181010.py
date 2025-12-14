import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Best for small-medium pooling windows (1-3)
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C': 64}, num_warps=4),
        # Best for medium pooling windows (3-5)
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C': 16}, num_warps=8),
        # Best for large pooling windows (5+)
        triton.Config({'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_W': 64, 'BLOCK_SIZE_C': 4}, num_warps=16),
        # Balanced config for various scenarios
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C': 32}, num_warps=8),
    ],
    key=['batch_size', 'channels', 'pooled_height', 'pooled_width', 'kernel_h', 'kernel_w'],
)
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
    # Grid dimensions for re-mapping
    grid_batch: tl.constexpr,
    grid_channel: tl.constexpr,
    # Block parameters - now autotuned
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize across batch, channels, and output spatial dimensions using 3D grid
    pid_ph = tl.program_id(0)
    pid_pw = tl.program_id(1)
    pid_combined = tl.program_id(2)
    
    # Split combined dimension into batch and channel blocks
    pid_batch = pid_combined // grid_channel
    pid_channel = pid_combined % grid_channel
    
    # Early exit for batch boundary
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
    
    # Initialize output with -inf using efficient full generation
    output_block = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), 
                          float('-inf'), dtype=tl.float32)
    
    # Pre-compute kernel offsets to reduce computations in inner loop
    kernel_h_offsets = tl.arange(0, kernel_h) * dilation_h
    kernel_w_offsets = tl.arange(0, kernel_w) * dilation_w
    
    # Main pooling loop
    for kh_idx in range(kernel_h):
        kh = kernel_h_offsets[kh_idx]
        for kw_idx in range(kernel_w):
            kw = kernel_w_offsets[kw_idx]
            
            # Compute input positions with dilation - optimized indexing
            h_idx = ph_offsets[:, None, None] * stride_h - pad_h + kh
            w_idx = pw_offsets[None, :, None] * stride_w - pad_w + kw
            
            # Create mask for valid input positions
            h_valid = (h_idx >= 0) & (h_idx < height)
            w_valid = (w_idx >= 0) & (w_idx < width)
            valid_mask = h_valid & w_valid  # [BLOCK_SIZE_H, BLOCK_SIZE_W, 1]
            
            # Compute input pointers with pre-computed strides
            x_ptrs = (
                x_ptr +
                pid_batch * batch_stride_x +
                channel_offsets[None, None, :] * channel_stride_x +
                h_idx * height_stride_x +
                w_idx * width_stride_x
            )
            
            # Load input values with boundary checking - optimized mask creation
            load_mask = valid_mask & channel_mask[None, None, :]
            x_vals = tl.load(x_ptrs, mask=load_mask, other=float('-inf'))
            
            # Update max (invalid positions are -inf, so they don't affect the max)
            output_block = tl.maximum(output_block, x_vals)
    
    # Store results with optimized pointer computation
    output_ptrs = (
        output_ptr +
        pid_batch * batch_stride_out +
        channel_offsets[None, None, :] * channel_stride_out +
        ph_offsets[:, None, None] * height_stride_out +
        pw_offsets[None, :, None] * width_stride_out
    )
    
    # Final output mask
    output_mask = (
        ph_mask[:, None, None] &
        pw_mask[None, :, None] &
        channel_mask[None, None, :]
    )
    
    tl.store(output_ptrs, output_block, mask=output_mask)


def triton_max_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Triton implementation of 2D max pooling with autotuning."""
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
    
    # Grid calculation will be handled by autotune configs
    # We launch with maximum possible grid and let autotune choose the best block sizes
    
    # Calculate grid dimensions (we'll compute these per config in the autotune)
    # Use dummy values for now, actual values computed per config
    
    # 3D grid: (ph_blocks, pw_blocks, batch*channel_blocks)
    # This exposes more independent parallelism to SMs
    grid = (
        triton.cdiv(pooled_height, 16),  # Conservative starting point
        triton.cdiv(pooled_width, 16),   # Conservative starting point
        batch_size * triton.cdiv(channels, 16),  # Conservative starting point
    )
    
    # Launch kernel - autotune will select the best config
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
        grid_batch=batch_size,
        grid_channel=triton.cdiv(channels, 16),  # Will be overridden by actual config
        # BLOCK_SIZE_* will be provided by autotune
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D using Triton kernels with autotuning.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window. If None, defaults to kernel_size.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
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
