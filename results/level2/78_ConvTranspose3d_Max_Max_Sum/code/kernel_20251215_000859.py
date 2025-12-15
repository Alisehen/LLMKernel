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
    D_out1, H_out1, W_out1,
    D_out2, H_out2, W_out2,
    # Kernel sizes
    K1: tl.constexpr,
    K2: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel for:
    1. MaxPool3d(kernel_size=2) -> reduces dimensions by factor of 2
    2. MaxPool3d(kernel_size=3) -> reduces dimensions by factor of 3
    3. Sum over channels (dim=1) with keepdim=True
    
    Input shape: [N, C, D, H, W]
    Intermediate after first pool: [N, C, D//2, H//2, W//2]
    Intermediate after second pool: [N, C, (D//2)//3, (H//2)//3, (W//2)//3]
    Output shape: [N, 1, D_out, H_out, W_out] where D_out = (D//2)//3
    
    Each program handles one output spatial position and one batch
    """
    # Compute program ID mapping to output position
    pid = tl.program_id(0)
    num_output_positions = D_out2 * H_out2 * W_out2
    
    n_idx = pid // num_output_positions
    pos_idx = pid % num_output_positions
    
    w2_idx = pos_idx % W_out2
    h2_idx = (pos_idx // W_out2) % H_out2
    d2_idx = pos_idx // (W_out2 * H_out2)
    
    # Check bounds
    if n_idx >= N:
        return
    
    # Compute start indices for the second maxpool window in intermediate space
    d1_start = d2_idx * K2
    h1_start = h2_idx * K2
    w1_start = w2_idx * K2
    
    # Compute start indices for the first maxpool window in original space
    # Each intermediate position comes from a K1xK1xK1 window
    d_start = d1_start * K1
    h_start = h1_start * K1
    w_start = w1_start * K1
    
    # Channel offsets for this block
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Initialize accumulator for sum over channels
    out_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Iterate over second maxpool window in intermediate space
    for kd2 in range(K2):
        for kh2 in range(K2):
            for kw2 in range(K2):
                d1 = d1_start + kd2
                h1 = h1_start + kh2
                w1 = w1_start + kw2
                
                # FIXED: Use explicit parentheses for boolean conditions
                if (d1 < D_out1) and (h1 < H_out1) and (w1 < W_out1):
                    # For each position in intermediate space, compute max over first pool window
                    max_vals = tl.full((BLOCK_C,), float('-inf'), dtype=tl.float32)
                    
                    # Iterate over first maxpool window in original space
                    for kd1 in range(K1):
                        for kh1 in range(K1):
                            for kw1 in range(K1):
                                d = d_start + kd1 + kd2 * K1
                                h = h_start + kh1 + kh2 * K1
                                w = w_start + kw1 + kw2 * K1
                                
                                # FIXED: Use explicit parentheses for boolean conditions
                                if (d < D) and (h < H) and (w < W):
                                    # Compute pointer offsets for all channels in block
                                    x_ptrs = (
                                        x_ptr +
                                        n_idx * stride_n +
                                        c_offsets * stride_c +
                                        d * stride_d +
                                        h * stride_h +
                                        w * stride_w
                                    )
                                    
                                    # Load values for all channels
                                    vals = tl.load(x_ptrs, mask=c_mask, other=float('-inf'))
                                    
                                    # Update max values
                                    max_vals = tl.maximum(max_vals, vals)
                    
                    # Accumulate max values for this position in second pool window
                    out_acc += tl.where(c_mask, max_vals, 0.0)
    
    # Sum over channels (dim=1) with keepdim=True
    channel_sum = tl.sum(out_acc, axis=0)
    
    # Compute output pointer
    out_idx = (
        n_idx * D_out2 * H_out2 * W_out2 +
        d2_idx * H_out2 * W_out2 +
        h2_idx * W_out2 +
        w2_idx
    )
    
    # Store result
    tl.store(out_ptr + out_idx, channel_sum)


def fused_double_maxpool_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Fused implementation of:
    1. MaxPool3d(kernel_size=2)
    2. MaxPool3d(kernel_size=3)
    3. Sum over channels (dim=1) with keepdim=True
    """
    N, C, D, H, W = x.shape
    
    # Output dimensions after first pool (kernel_size=2, stride=2 by default)
    D_out1 = D // 2
    H_out1 = H // 2
    W_out1 = W // 2
    
    # Output dimensions after second pool (kernel_size=3, stride=3 by default)
    D_out2 = D_out1 // 3
    H_out2 = H_out1 // 3
    W_out2 = W_out1 // 3
    
    # Create output tensor [N, 1, D_out2, H_out2, W_out2]
    out = torch.empty((N, 1, D_out2, H_out2, W_out2),
                      device=x.device, dtype=x.dtype)
    
    # Calculate grid size - one program per output spatial position per batch
    grid = (N * D_out2 * H_out2 * W_out2,)
    
    # Choose block size for channels (power of 2)
    BLOCK_C = triton.next_power_of_2(C)
    
    # Launch kernel
    fused_double_maxpool_sum_kernel[grid](
        x,
        out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        D_out1, H_out1, W_out1,
        D_out2, H_out2, W_out2,
        K1=2,  # First maxpool kernel size
        K2=3,  # Second maxpool kernel size
        BLOCK_C=BLOCK_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    ConvTranspose3d is kept in PyTorch native, only the pooling and sum operations are fused in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native - DO NOT reimplement in Triton
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
