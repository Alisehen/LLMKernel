import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit(
    num_warps=lambda args: args['BLOCK_C'] // 32,
    num_stages=lambda args: args['NUM_STAGES']
)
def fused_double_maxpool_sum_kernel(
    x_ptr,
    out_ptr,
    # Tensor dimensions
    N, C, D, H, W,
    # Strides for input
    stride_n, stride_c, stride_d, stride_h, stride_w,
    # Output dimensions after pooling
    D_out1, H_out1, W_out1,
    D_out2, H_out2, W_out2,
    # Kernel sizes and optimization params
    K1: tl.constexpr,
    K2: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """
    Optimized fused kernel for:
    1. MaxPool3d(kernel_size=2)
    2. MaxPool3d(kernel_size=3)
    3. Sum over channels (dim=1) with keepdim=True
    """
    # Compute program ID - 1D grid over all output positions
    pid = tl.program_id(0)
    num_output_positions = D_out2 * H_out2 * W_out2
    
    n_idx = pid // num_output_positions
    pos_idx = pid % num_output_positions
    
    # Convert linear position to 3D coordinates
    w2_idx = pos_idx % W_out2
    h2_idx = (pos_idx // W_out2) % H_out2
    d2_idx = pos_idx // (W_out2 * H_out2)
    
    # Early exit if batch index out of bounds
    if n_idx >= N:
        return
    
    # Compute window start positions with pre-multiplied strides
    d1_start = d2_idx * K2
    h1_start = h2_idx * K2
    w1_start = w2_idx * K2
    
    d_start = d1_start * K1
    h_start = h1_start * K1
    w_start = w1_start * K1
    
    # Channel offsets for vectorized processing
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Precompute base offset for this batch element
    batch_offset = n_idx * stride_n
    
    # Initialize accumulator with proper dtype for TF32
    if USE_TF32:
        out_acc = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)
    else:
        out_acc = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)
    
    # Unroll K2 loop for better performance
    for kd2 in tl.static_unroll(range(K2)):
        d1 = d1_start + kd2
        d1_in_bounds = d1 < D_out1
        
        if d1_in_bounds:
            for kh2 in tl.static_unroll(range(K2)):
                h1 = h1_start + kh2
                h1_in_bounds = h1 < H_out1
                
                if h1_in_bounds:
                    for kw2 in tl.static_unroll(range(K2)):
                        w1 = w1_start + kw2
                        w1_in_bounds = w1 < W_out1
                        
                        if w1_in_bounds:
                            # Initialize max values for first pool
                            max_vals = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)
                            
                            # Unroll K1 loop completely (small fixed size)
                            for kd1 in tl.static_unroll(range(K1)):
                                d = d_start + kd1 + kd2 * K1
                                d_in_bounds = d < D
                                
                                if d_in_bounds:
                                    for kh1 in tl.static_unroll(range(K1)):
                                        h = h_start + kh1 + kh2 * K1
                                        h_in_bounds = h < H
                                        
                                        if h_in_bounds:
                                            for kw1 in tl.static_unroll(range(K1)):
                                                w = w_start + kw1 + kw2 * K1
                                                w_in_bounds = w < W
                                                
                                                if w_in_bounds:
                                                    # Compute pointer with fused offset calculation
                                                    offset = (
                                                        batch_offset +
                                                        d * stride_d +
                                                        h * stride_h +
                                                        w * stride_w
                                                    )
                                                    x_ptrs = x_ptr + offset + c_offsets * stride_c
                                                    
                                                    # Load values for all channels
                                                    vals = tl.load(x_ptrs, mask=c_mask, other=float('-inf'))
                                                    
                                                    # Update max values
                                                    max_vals = tl.maximum(max_vals, vals)
                            
                            # Update second pool max
                            out_acc = tl.maximum(out_acc, max_vals)
    
    # Sum over channels using tree reduction for better precision
    channel_sum = tl.sum(out_acc, axis=0)
    
    # Store result
    out_idx = n_idx * num_output_positions + pos_idx
    tl.store(out_ptr + out_idx, channel_sum)


def fused_double_maxpool_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation with autotuned configurations.
    """
    N, C, D, H, W = x.shape
    
    # Output dimensions
    D_out1 = D // 2
    H_out1 = H // 2
    W_out1 = W // 2
    
    D_out2 = D_out1 // 3
    H_out2 = H_out1 // 3
    W_out2 = W_out1 // 3
    
    # Create output tensor
    out = torch.empty((N, 1, D_out2, H_out2, W_out2),
                      device=x.device, dtype=x.dtype)
    
    # Grid size
    grid = (N * D_out2 * H_out2 * W_out2,)
    
    # Determine optimal block size based on channel count
    # Use powers of 2 but cap at 256 to avoid register pressure
    if C <= 64:
        BLOCK_C = 64
    elif C <= 128:
        BLOCK_C = 128
    elif C <= 256:
        BLOCK_C = 256
    else:
        BLOCK_C = 256  # Maximum for good occupancy
    
    # Choose configuration based on channel count and memory pressure
    if C <= 64:
        # Low register pressure - can use aggressive settings
        num_stages = 3
        num_warps = BLOCK_C // 32
    elif C <= 128:
        # Moderate pressure
        num_stages = 2
        num_warps = BLOCK_C // 32
    else:
        # High pressure (C > 128)
        num_stages = 2
        # Reduce warps for better register allocation
        if BLOCK_C == 256:
            num_warps = 6  # 192 threads, better register distribution
        else:
            num_warps = BLOCK_C // 32
    
    # Enable TF32 on Ada Lovelace for better throughput
    USE_TF32 = True if x.dtype in [torch.float32, torch.bfloat16] else False
    
    # Launch optimized kernel
    fused_double_maxpool_sum_kernel[grid](
        x,
        out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        D_out1, H_out1, W_out1,
        D_out2, H_out2, W_out2,
        K1=2,
        K2=3,
        BLOCK_C=BLOCK_C,
        NUM_STAGES=num_stages,
        USE_TF32=USE_TF32,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
    
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused double maxpool + sum in Triton
        x = fused_double_maxpool_sum(x)
        
        return x
