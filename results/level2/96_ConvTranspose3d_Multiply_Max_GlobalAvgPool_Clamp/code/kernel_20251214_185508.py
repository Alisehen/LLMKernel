import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_post_ops_kernel(
    x_ptr,                 # Input tensor pointer
    out_ptr,               # Output tensor pointer
    N, C, D, H, W,         # Input dimensions
    scale,                 # Scalar multiplier
    K,                     # MaxPool kernel size
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    """Fused kernel: Scale + MaxPool3d + GlobalAvgPool + Clamp"""
    pid_n = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Channel index (handles BLOCK_C channels)
    
    # Early exit for out-of-bounds programs
    if pid_n >= N:
        return
    
    # Process BLOCK_C channels (or remaining ones)
    channel_base = pid_c * BLOCK_C
    channel_mask = channel_base + tl.arange(0, BLOCK_C) < C
    
    # Compute max pooling output dimensions
    D_out = D // K
    H_out = H // K
    W_out = W // K
    total_windows = D_out * H_out * W_out
    
    # Initialize accumulators for global average pooling
    gap_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    valid_counts = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Loop over max pooling output windows
    for window_idx in range(total_windows):
        d_out = window_idx // (H_out * W_out)
        h_out = (window_idx % (H_out * W_out)) // W_out
        w_out = window_idx % W_out
        
        # Initialize max values and validity flags for this window
        max_vals = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)
        window_has_valid = tl.zeros((BLOCK_C,), dtype=tl.int1)
        
        # Compute max over K³ window
        for kd in range(K):
            for kh in range(K):
                for kw in range(K):
                    d_in = d_out * K + kd
                    h_in = h_out * K + kh
                    w_in = w_out * K + kw
                    
                    # Check if this position is within bounds
                    in_bounds = (d_in < D) & (h_in < H) & (w_in < W)
                    
                    if tl.any(in_bounds):
                        # Load input values for all channels in block
                        for c_offset in range(BLOCK_C):
                            c_idx = channel_base + c_offset
                            if c_idx < C and channel_mask[c_offset]:
                                x_ptr_base = x_ptr + pid_n * stride_xn + c_idx * stride_xc
                                x_val = tl.load(
                                    x_ptr_base + d_in * stride_xd + h_in * stride_xh + w_in * stride_xw,
                                    mask=in_bounds,
                                    other=0.0
                                )
                                x_val = x_val * scale
                                
                                # Update max and validity flag
                                if in_bounds:
                                    old_max = tl.load(max_vals + c_offset)
                                    new_max = tl.maximum(old_max, x_val)
                                    tl.store(max_vals + c_offset, new_max)
                                    tl.store(window_has_valid + c_offset, True)
        
        # Accumulate max values for valid windows
        for c_offset in range(BLOCK_C):
            if channel_mask[c_offset]:
                has_valid = tl.load(window_has_valid + c_offset)
                if has_valid:
                    max_val = tl.load(max_vals + c_offset)
                    tl.store(gap_acc + c_offset, tl.load(gap_acc + c_offset) + max_val)
                    tl.store(valid_counts + c_offset, tl.load(valid_counts + c_offset) + 1.0)
    
    # Compute global average and clamp for each channel in block
    for c_offset in range(BLOCK_C):
        c_idx = channel_base + c_offset
        if c_idx < C and channel_mask[c_offset]:
            valid_count = tl.load(valid_counts + c_offset)
            if valid_count > 0:
                avg_val = tl.load(gap_acc + c_offset) / valid_count
            else:
                avg_val = 0.0
            
            # Clamp between 0 and 1
            avg_val = tl.minimum(tl.maximum(avg_val, 0.0), 1.0)
            
            # Store result [N, C, 1, 1, 1]
            out_idx = pid_n * stride_on + c_idx * stride_oc
            tl.store(out_ptr + out_idx, avg_val)


def fused_post_convtranspose(x, scale, maxpool_kernel_size):
    """
    Fused operations: Scale + MaxPool3d + GlobalAvgPool + Clamp
    Input: [N, C, D, H, W] -> Output: [N, C, 1, 1, 1]
    """
    N, C, D, H, W = x.shape
    K = maxpool_kernel_size
    
    # Output tensor: [N, C, 1, 1, 1]
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    # Choose block size for channels (power of 2)
    BLOCK_C = min(triton.next_power_of_2(C), 16)  # Reduced for better occupancy
    
    # Grid configuration
    grid = (N, triton.cdiv(C, BLOCK_C))
    
    # Launch kernel
    fused_post_ops_kernel[grid](
        x, out,
        N, C, D, H, W,
        scale, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused post-ops (Triton)
    Uses PyTorch's ConvTranspose3d and fuses Scale + MaxPool3d + GlobalAvgPool + Clamp
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native - DO NOT reimplement in Triton
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        # Remove individual pooling layers since they're fused in Triton
        
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        # Step 2: Fused post-ops in Triton
        x = fused_post_convtranspose(x, self.scale, self.maxpool_kernel_size)
        return x
