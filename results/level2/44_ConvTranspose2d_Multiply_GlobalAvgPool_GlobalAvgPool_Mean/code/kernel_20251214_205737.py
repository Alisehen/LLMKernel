import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_multiply_gap_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    multiplier,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel: Multiply scalar + Global Average Pooling over H,W
    Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    """
    pid_n = tl.program_id(0)  # Batch dimension
    pid_c = tl.program_id(1)  # Channel block index
    
    if pid_n >= N:
        return
    
    # Channel offsets for this block
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Initialize accumulation for each channel in this block
    channel_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Process spatial dimensions in tiles
    hw_size = H * W
    for hw_start in range(0, hw_size, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_size
        
        # Convert 1D spatial offset to 2D (h, w)
        h = hw_offsets // W
        w = hw_offsets % W
        
        # Load and process tile for all channels in block
        # Compute pointer offsets for all channels in block
        x_base = pid_n * stride_xn + c_offsets[:, None] * stride_xc
        
        # Compute spatial offsets (broadcast across channels)
        h_offsets = h[None, :] * stride_xh
        w_offsets = w[None, :] * stride_xw
        
        # Combined pointer offsets for all (channel, spatial) combinations
        ptr_offsets = x_base + h_offsets + w_offsets  # Shape: [BLOCK_C, BLOCK_HW]
        
        # Create 2D mask for valid (channel, spatial) positions
        mask_2d = c_mask[:, None] & hw_mask[None, :]
        
        # Load data block with masking
        block = tl.load(x_ptr + ptr_offsets, mask=mask_2d, other=0.0)
        
        # Accumulate over spatial dimension
        channel_sum += tl.sum(block, axis=1)
    
    # Compute average and apply multiplier
    spatial_size = H * W
    channel_avg = channel_sum / spatial_size
    result = channel_avg * multiplier
    
    # Store results at position (n, c, 0, 0)
    out_base = pid_n * stride_on + c_offsets * stride_oc
    tl.store(out_ptr + out_base, result, mask=c_mask)


def fused_multiply_gap(x, multiplier):
    """
    Fused: Multiply scalar + Global Average Pooling
    Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    
    # Output tensor with shape [N, C, 1, 1]
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Choose block sizes as powers of 2
    # Limit BLOCK_C to improve occupancy
    BLOCK_C = min(triton.next_power_of_2(C), 128)
    # Adjust BLOCK_HW based on available shared memory and register pressure
    max_hw = min(H * W, 1024 // BLOCK_C)  # Heuristic for better performance
    BLOCK_HW = min(triton.next_power_of_2(max_hw), 256)
    
    # Grid: (N, number of channel blocks)
    grid = (N, triton.cdiv(C, BLOCK_C))
    
    # Launch kernel
    fused_multiply_gap_kernel[grid](
        x, out, multiplier,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
    )
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + Fused multiply + GAP
    
    NOTE: ConvTranspose2d uses PyTorch's native implementation.
    Only the scalar multiplication and global average pooling are fused in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 output_padding, multiplier):
        super(ModelNew, self).__init__()
        # ConvTranspose2d remains PyTorch native
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.multiplier = multiplier
    
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose2d
        x = self.conv_transpose(x)
        # Step 2: Fused scalar multiplication + global average pooling
        x = fused_multiply_gap(x, self.multiplier)
        # Second GAP is redundant (input is already 1x1)
        return x
