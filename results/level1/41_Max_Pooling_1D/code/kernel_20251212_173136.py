import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_forward_kernel(
    x_ptr,
    output_ptr,
    B,
    C,
    L_in,
    L_out,
    kernel_size,
    stride,
    padding,
    dilation,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # 2D grid: (C * triton.cdiv(L_out, BLOCK_SIZE_IN), B)
    pid_c_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Decompose first dimension
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_out = (L_out + BLOCK_SIZE_IN - 1) // BLOCK_SIZE_IN
    pid_c_block = pid_c_out // grid_out
    pid_out_block = pid_c_out % grid_out
    
    # Channel offsets
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Output position offsets
    out_start = pid_out_block * BLOCK_SIZE_IN
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE_IN)
    out_mask = out_offsets < L_out
    
    # Initialize max values
    max_vals = tl.full((BLOCK_SIZE_C, BLOCK_SIZE_IN), float('-inf'), dtype=tl.float32)
    
    # Process kernel positions sequentially (unrolled at compile time)
    # Use tl.range for Triton-style loop
    kernel_range = tl.arange(0, kernel_size)
    
    # For each kernel position
    for k_idx in range(kernel_size):
        # Calculate input positions for all output positions in block
        input_positions = out_offsets * stride - padding + k_idx * dilation
        
        # Check input bounds
        input_valid = (input_positions >= 0) & (input_positions < L_in)
        
        # Load input values for all channels in vectorized way
        for c_idx in range(BLOCK_SIZE_C):
            if c_mask[c_idx]:
                channel = c_start + c_idx
                # Base offset for this batch and channel
                base_offset = pid_b * C * L_in + channel * L_in
                
                # Load values for all output positions at once
                # Create mask for valid positions
                load_mask = out_mask & input_valid
                input_offsets = base_offset + input_positions
                values = tl.load(x_ptr + input_offsets, mask=load_mask, other=float('-inf'))
                
                # Update max for this channel
                # Need to handle updating a specific row in max_vals
                # We'll create a mask for this channel
                channel_mask = tl.arange(0, BLOCK_SIZE_C) == c_idx
                # Expand to match BLOCK_SIZE_IN
                channel_mask_2d = channel_mask[:, None]
                
                # Get current max for this channel
                current_max = tl.where(channel_mask_2d, max_vals, float('-inf'))
                current_max_row = tl.sum(current_max, axis=0)  # Reduce to single row
                
                # Compute new max
                new_max_row = tl.maximum(current_max_row, values)
                
                # Update max_vals for this channel
                # Create update mask
                update_mask = channel_mask_2d & (out_mask[None, :])
                # Create new values matrix
                new_vals = tl.where(channel_mask_2d, new_max_row[None, :], max_vals)
                max_vals = tl.where(update_mask, new_vals, max_vals)
    
    # Store results
    for c_idx in range(BLOCK_SIZE_C):
        if c_mask[c_idx]:
            channel = c_start + c_idx
            output_offset = pid_b * C * L_out + channel * L_out + out_offsets
            tl.store(output_ptr + output_offset, max_vals[c_idx, :], mask=out_mask)


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False
) -> torch.Tensor:
    """
    Max Pooling 1D implemented with Triton kernels.
    
    Args:
        x: Input tensor of shape (B, C, L_in)
        kernel_size: Size of the window to take max over
        stride: Stride of the window (defaults to kernel_size)
        padding: Zero padding added to both sides
        dilation: Spacing between kernel elements
        return_indices: Whether to return indices (not implemented)
    
    Returns:
        Output tensor of shape (B, C, L_out)
    """
    if stride is None:
        stride = kernel_size
    
    B, C, L_in = x.shape
    
    # Calculate output length
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(B, C, L_out, device=x.device, dtype=x.dtype)
    
    # Choose optimal block sizes
    BLOCK_SIZE_IN = 64
    BLOCK_SIZE_C = 32
    
    # Launch kernel
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_out = (L_out + BLOCK_SIZE_IN - 1) // BLOCK_SIZE_IN
    
    grid = (grid_c * grid_out, B)
    
    maxpool1d_forward_kernel[grid](
        x,
        output,
        B,
        C,
        L_in,
        L_out,
        kernel_size,
        stride,
        padding,
        dilation,
        BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D with optimized Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor using Triton kernels.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, sequence_length).
        
        Returns:
            Output tensor with Max Pooling 1D applied.
        """
        return triton_maxpool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices
        )
