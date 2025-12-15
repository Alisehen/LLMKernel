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
    # Block sizes
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
    """
    # 3D grid mapping
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    # Check batch bounds
    if pid_n >= N:
        return
    
    # Calculate spatial offsets
    d_base = pid_d * BLOCK_D
    h_base = pid_h * BLOCK_H
    
    # Initialize accumulators with proper dimensions
    out_acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Channel block processing
    for c_block_start in range(0, C, BLOCK_C):
        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        
        # Initialize channel max accumulators
        channel_max = tl.full((BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_C), 
                            float('-inf'), dtype=tl.float32)
        
        # Process each output position in the block
        for d_idx in tl.range(0, BLOCK_D):
            for h_idx in tl.range(0, BLOCK_H):
                for w_idx in tl.range(0, BLOCK_W):
                    # Calculate output spatial position
                    d_out = d_base + d_idx
                    h_out = h_base + h_idx
                    w_out = w_idx
                    
                    # Skip if out of bounds
                    if d_out >= D_out or h_out >= H_out or w_out >= W_out:
                        continue
                    
                    # Calculate input window start for combined pooling (6 = 2*3)
                    d_start = d_out * 6
                    h_start = h_out * 6
                    w_start = w_out * 6
                    
                    # Combined max over 6x6x6 window
                    for kd in range(6):
                        d_in = d_start + kd
                        if d_in >= D:
                            continue
                        for kh in range(6):
                            h_in = h_start + kh
                            if h_in >= H:
                                continue
                            for kw in range(6):
                                w_in = w_start + kw
                                if w_in >= W:
                                    continue
                                
                                # Compute pointer offsets
                                x_ptrs = (
                                    x_ptr +
                                    pid_n * stride_n +
                                    c_offsets[:, None] * stride_c +
                                    d_in * stride_d +
                                    h_in * stride_h +
                                    w_in * stride_w
                                )
                                
                                # Load values with mask
                                vals = tl.load(x_ptrs, mask=c_mask[:, None], 
                                              other=float('-inf'))
                                
                                # Update max
                                for c_idx in range(BLOCK_C):
                                    if c_mask[c_idx]:
                                        current = channel_max[d_idx, h_idx, w_idx, c_idx]
                                        channel_max = tl.where(
                                            (d_idx == d_idx) & (h_idx == h_idx) & 
                                            (w_idx == w_idx) & (c_idx == c_idx),
                                            tl.maximum(current, vals[c_idx]),
                                            channel_max
                                        )
        
        # Reduce across channels for this block
        block_sum = tl.sum(channel_max, axis=3)
        out_acc += block_sum
    
    # Store results
    for d_idx in range(BLOCK_D):
        d_out = d_base + d_idx
        if d_out >= D_out:
            continue
        for h_idx in range(BLOCK_H):
            h_out = h_base + h_idx
            if h_out >= H_out:
                continue
            for w_idx in range(BLOCK_W):
                w_out = w_idx
                if w_out >= W_out:
                    continue
                
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
    D_out = D // 6
    H_out = H // 6
    W_out = W // 6
    
    # Create output tensor
    out = torch.empty((N, 1, D_out, H_out, W_out),
                      device=x.device, dtype=x.dtype)
    
    # Optimized block sizes
    BLOCK_C = min(triton.next_power_of_2(C), 256)
    BLOCK_D = 4
    BLOCK_H = 4
    BLOCK_W = 16
    
    # 3D grid
    grid = (
        N,  # Process all batches in parallel
        triton.cdiv(D_out, BLOCK_D),
        triton.cdiv(H_out, BLOCK_H)
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
        num_warps=4,
        num_stages=3,
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
