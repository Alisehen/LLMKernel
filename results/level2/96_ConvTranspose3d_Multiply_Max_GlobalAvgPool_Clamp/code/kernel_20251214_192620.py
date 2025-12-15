import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_maxpool_avg_clamp_kernel(
    x_ptr,                 # Input tensor pointer
    out_ptr,              # Output tensor pointer [N, C, 1, 1, 1]
    N, C, D, H, W,        # Input dimensions
    scale,                # Scalar multiplier
    K,                    # MaxPool kernel size
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,  # Spatial tiles per thread
    num_warps: tl.constexpr,
):
    """
    Fused kernel: Scale + MaxPool3d + GlobalAvgPool + Clamp
    - Handles non-divisible dimensions with proper masking
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    # Channel block processing
    channel_base = pid_c * BLOCK_C
    channel_offsets = tl.arange(0, BLOCK_C)
    channel_mask = channel_base + channel_offsets < C
    
    # Output dimensions for maxpool (ceil division)
    D_out = (D + K - 1) // K
    H_out = (H + K - 1) // K
    W_out = (W + K - 1) // K
    total_windows = D_out * H_out * W_out
    
    # Initialize accumulators in registers
    sum_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Process spatial windows in tiles
    for window_start in range(0, total_windows, BLOCK_W):
        window_offsets = window_start + tl.arange(0, BLOCK_W)
        window_mask = window_offsets < total_windows
        
        if tl.sum(window_mask) > 0:
            # Convert linear window index to 3D indices
            d_out = window_offsets // (H_out * W_out)
            h_out = (window_offsets % (H_out * W_out)) // W_out
            w_out = window_offsets % W_out
            
            # Initialize max values for this window tile
            window_max = tl.full((BLOCK_C, BLOCK_W), -float('inf'), dtype=tl.float32)
            
            # Compute max over K³ window for each spatial position
            for kd in range(K):
                d_in = d_out * K + kd
                d_in_valid = d_in < D
                d_offset = d_in * stride_xd
                
                for kh in range(K):
                    h_in = h_out * K + kh
                    h_in_valid = h_in < H
                    h_offset = h_in * stride_xh
                    
                    for kw in range(K):
                        w_in = w_out * K + kw
                        w_in_valid = w_in < W
                        w_offset = w_in * stride_xw
                        
                        # Compute load pointers with broadcasting
                        base_ptr = (
                            x_ptr + 
                            pid_n * stride_xn + 
                            channel_base * stride_xc
                        )
                        
                        ptr_offsets = (
                            channel_offsets[:, None] * stride_xc +
                            d_offset[None, :] +
                            h_offset[None, :] +
                            w_offset[None, :]
                        )
                        
                        x_ptrs = base_ptr + ptr_offsets
                        
                        # Combined mask: channel, window, and spatial validity
                        load_mask = (
                            channel_mask[:, None] & 
                            window_mask[None, :] &
                            d_in_valid[None, :] &
                            h_in_valid[None, :] &
                            w_in_valid[None, :]
                        )
                        
                        # Load and scale (use 0.0 as other for max computation)
                        vals = tl.load(x_ptrs, mask=load_mask, other=-float('inf'))
                        vals = vals * scale
                        
                        # Update max (only where mask is valid)
                        window_max = tl.where(load_mask, tl.maximum(window_max, vals), window_max)
            
            # Accumulate max values for average
            valid_mask = channel_mask[:, None] & window_mask[None, :]
            window_max_valid = tl.where(valid_mask, window_max, 0.0)
            
            # Reduce over spatial dimension
            sum_acc += tl.sum(window_max_valid, axis=1)
            count_acc += tl.sum(valid_mask.to(tl.float32), axis=1)
    
    # Compute average and clamp
    avg_vals = tl.where(count_acc > 0, sum_acc / count_acc, 0.0)
    avg_vals = tl.minimum(tl.maximum(avg_vals, 0.0), 1.0)
    
    # Store final result
    out_ptrs = (
        out_ptr + 
        pid_n * stride_on + 
        (channel_base + channel_offsets) * stride_oc
    )
    tl.store(out_ptrs, avg_vals, mask=channel_mask)


def fused_post_convtranspose(x, scale, maxpool_kernel_size):
    """
    Optimized fused operations using a single kernel:
    1. Scale + MaxPool3d
    2. GlobalAvgPool + Clamp
    
    Eliminates intermediate tensor stores completely.
    """
    N, C, D, H, W = x.shape
    K = maxpool_kernel_size
    
    # Output tensor
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    # Tuning parameters optimized for Ada Lovelace
    # - BLOCK_C: Maximize channel parallelism within register limits
    # - BLOCK_W: Balance spatial parallelism and register usage
    # - num_warps: Match SM occupancy (1536 threads/SM ÷ 32 threads/warp = 48 warps max)
    BLOCK_C = 64  # Increased for better SM occupancy
    BLOCK_W = 64  # Optimized for spatial parallelism
    num_warps = 8  # 256 threads, leaves room for concurrent warps on SM
    
    grid = (N, triton.cdiv(C, BLOCK_C))
    
    fused_scale_maxpool_avg_clamp_kernel[grid](
        x, out,
        N, C, D, H, W,
        scale, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Optimized fused post-ops (Triton)
    Uses a single fused kernel to eliminate all intermediate stores
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size
        
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        # Step 2: Optimized fused post-ops in Triton (single kernel)
        x = fused_post_convtranspose(x, self.scale, self.maxpool_kernel_size)
        return x
