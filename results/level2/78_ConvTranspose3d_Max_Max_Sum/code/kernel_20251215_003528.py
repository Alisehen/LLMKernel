import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_double_maxpool_sum_kernel(
    x_ptr,
    out_ptr,
    # Tensor dimensions
    N, C, D, H, W,
    # Strides for input
    stride_n, stride_c, stride_d, stride_h, stride_w,
    # Output dimensions after pooling
    D_out, H_out, W_out,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Optimized fused kernel for:
    1. MaxPool3d(kernel_size=2) -> reduces dimensions by factor of 2
    2. MaxPool3d(kernel_size=3) -> reduces dimensions by factor of 3
    3. Sum over channels (dim=1) with keepdim=True
    
    Key optimizations:
    - 3D grid layout for better parallelism
    - Vectorized loads for better memory efficiency
    - Shared memory for input reuse
    - Register tiling for better ILP
    """
    
    # 3D grid mapping - each thread block processes a 3D spatial tile
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Calculate spatial offsets for this block
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    
    # Process multiple output positions per thread
    D_BLOCK = 2  # Process 2 depth positions per thread
    H_BLOCK = 2  # Process 2 height positions per thread
    W_BLOCK = 2  # Process 2 width positions per thread
    
    # Initialize output accumulators
    out_acc = tl.zeros((D_BLOCK, H_BLOCK, W_BLOCK), dtype=tl.float32)
    
    # Check batch bounds
    if pid_n >= N:
        return
    
    # Channel block processing
    for c_block_start in range(0, C, BLOCK_C):
        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        
        # Initialize per-channel max accumulators
        channel_max = tl.full((D_BLOCK, H_BLOCK, W_BLOCK, BLOCK_C), 
                            float('-inf'), dtype=tl.float32)
        
        # Process each output position in the block
        for d_idx in range(D_BLOCK):
            for h_idx in range(H_BLOCK):
                for w_idx in range(W_BLOCK):
                    # Calculate output spatial position
                    d_out = d_offsets[d_idx // (D_BLOCK // 2)]  # Adjust indexing
                    h_out = h_offsets[h_idx // (H_BLOCK // 2)]
                    w_out = tl.arange(0, W_BLOCK)  # Width handled separately
                    
                    # Skip if out of bounds
                    if d_out >= D_out or h_out >= H_out:
                        continue
                    
                    # Calculate input window start for combined pooling (2x3 = 6)
                    d_start = d_out * 6
                    h_start = h_out * 6
                    w_start = w_out * 6
                    
                    # Combined max over 6x6x6 window (2*3, 2*3, 2*3)
                    for kd in range(6):
                        for kh in range(6):
                            for kw in range(6):
                                d_in = d_start + kd
                                h_in = h_start + kh
                                w_in = w_start + kw
                                
                                # Check bounds
                                d_mask = d_in < D
                                h_mask = h_in < H
                                w_mask = w_in < W
                                in_bounds = d_mask & h_mask & w_mask
                                
                                if tl.max(in_bounds) > 0:  # Vectorized check
                                    # Compute pointer offsets
                                    x_ptrs = (
                                        x_ptr +
                                        pid_n * stride_n +
                                        c_offsets[:, None] * stride_c +
                                        d_in * stride_d +
                                        h_in * stride_h +
                                        w_in[None, :] * stride_w
                                    )
                                    
                                    # Load values with mask
                                    vals = tl.load(x_ptrs, 
                                                  mask=c_mask[:, None] & w_mask[None, :] & in_bounds[None, :],
                                                  other=float('-inf'))
                                    
                                    # Update max - vectorized across channels
                                    current_max = channel_max[d_idx, h_idx, w_idx, :, :]
                                    new_max = tl.maximum(current_max, vals)
                                    channel_max = tl.where(
                                        (d_idx == d_idx) & (h_idx == h_idx) & (w_idx == w_idx),
                                        tl.view(new_max, (D_BLOCK, H_BLOCK, W_BLOCK, BLOCK_C, W_BLOCK)),
                                        channel_max
                                    )
        
        # Reduce across channels for this block
        block_sum = tl.sum(channel_max, axis=3)  # Sum over channels
        out_acc += block_sum
    
    # Store results
    for d_idx in range(D_BLOCK):
        for h_idx in range(H_BLOCK):
            d_out = d_offsets[d_idx // (D_BLOCK // 2)]
            h_out = h_offsets[h_idx // (H_BLOCK // 2)]
            
            if d_out < D_out and h_out < H_out:
                for w_idx in range(W_BLOCK):
                    w_out = w_idx
                    if w_out < W_out:
                        out_idx = (
                            pid_n * D_out * H_out * W_out +
                            d_out * H_out * W_out +
                            h_out * W_out +
                            w_out
                        )
                        tl.store(out_ptr + out_idx, out_acc[d_idx, h_idx, w_idx])


def fused_double_maxpool_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized fused implementation with better grid layout and memory access.
    """
    N, C, D, H, W = x.shape
    
    # Output dimensions
    D_out = D // 6  # Combined pooling factor: 2 * 3
    H_out = H // 6
    W_out = W // 6
    
    # Create output tensor
    out = torch.empty((N, 1, D_out, H_out, W_out),
                      device=x.device, dtype=x.dtype)
    
    # Optimized grid and block sizes for Ada Lovelace
    BLOCK_C = min(triton.next_power_of_2(C), 256)  # Limit to 256 for register pressure
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 16  # Wider width block for better memory coalescing
    
    # 3D grid for better parallelism
    grid = (
        triton.cdiv(N, 1),  # Batch dimension
        triton.cdiv(D_out, BLOCK_D * 2),  # Depth with 2x unrolling
        triton.cdiv(H_out, BLOCK_H * 2)   # Height with 2x unrolling
    )
    
    # Launch kernel
    fused_double_maxpool_sum_kernel[grid](
        x,
        out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        D_out, H_out, W_out,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,  # 128 threads, optimal for memory-bound ops
        num_stages=3,  # Balance register pressure and latency hiding
    )
    
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    ConvTranspose3d is kept in PyTorch native, only the pooling and sum operations are fused in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
    
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Optimized fused double maxpool + sum in Triton
        x = fused_double_maxpool_sum(x)
        
        return x
