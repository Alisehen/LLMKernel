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
        
        # Load and process tile
        for i in range(BLOCK_C):
            if c_mask[i]:
                # Compute pointer offsets
                c_offset = c_offsets[i]
                x_base = pid_n * stride_xn + c_offset * stride_xc
                
                # Accumulate over spatial positions in this tile
                spatial_sum = 0.0
                for j in range(BLOCK_HW):
                    if hw_mask[j]:
                        x_ptr_offset = x_base + h[j] * stride_xh + w[j] * stride_xw
                        val = tl.load(x_ptr + x_ptr_offset)
                        spatial_sum += val
                
                channel_sum = tl.where(
                    c_mask[i],
                    channel_sum + spatial_sum,
                    channel_sum
                )
    
    # Compute average and apply multiplier
    channel_avg = channel_sum / (H * W)
    result = channel_avg * multiplier
    
    # Store results at position (n, c, 0, 0)
    for i in range(BLOCK_C):
        if c_mask[i]:
            c_offset = c_offsets[i]
            out_ptr_offset = (pid_n * stride_on + 
                            c_offset * stride_oc + 
                            0 * stride_oh + 
                            0 * stride_ow)
            tl.store(out_ptr + out_ptr_offset, result[i])


def fused_multiply_gap(x, multiplier):
    """
    Fused: Multiply scalar + Global Average Pooling
    Input: [N, C, H, W] -> Output: [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    
    # Output tensor with shape [N, C, 1, 1]
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Choose block sizes as powers of 2
    BLOCK_C = triton.next_power_of_2(min(C, 128))
    BLOCK_HW = triton.next_power_of_2(min(H * W, 256))
    
    # Grid: (N, number of channel blocks)
    grid = (N, triton.cdiv(C, BLOCK_C))
    
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
