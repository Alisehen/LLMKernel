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
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Fused kernel: Scale + MaxPool3d + GlobalAvgPool + Clamp"""
    pid_n = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Channel index
    
    if pid_n >= N or pid_c >= C:
        return
    
    # Compute max pooling output dimensions
    D_out = D // K
    H_out = H // K
    W_out = W // K
    
    # Initialize accumulators for global average pooling
    gap_acc = tl.zeros((BLOCK_D * BLOCK_H * BLOCK_W,), dtype=tl.float32)
    valid_count = 0
    
    # Loop over max pooling output blocks
    for d_out in range(D_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Initialize max value for this pooling window
                max_val = tl.full((1,), -float('inf'), dtype=tl.float32)
                
                # Compute max over K³ window
                for kd in range(K):
                    for kh in range(K):
                        for kw in range(K):
                            d_in = d_out * K + kd
                            h_in = h_out * K + kh
                            w_in = w_out * K + kw
                            
                            # Load input value with scale multiplication
                            x_ptr_base = x_ptr + pid_n * stride_xn + pid_c * stride_xc
                            x_val = tl.load(
                                x_ptr_base + d_in * stride_xd + h_in * stride_xh + w_in * stride_xw,
                                mask=(d_in < D) & (h_in < H) & (w_in < W),
                                other=-float('inf')
                            )
                            x_val = x_val * scale
                            
                            # Update max
                            max_val = tl.maximum(max_val, x_val)
                
                # Store max value for global average pooling
                idx = d_out * (H_out * W_out) + h_out * W_out + w_out
                gap_acc = tl.where(
                    idx < BLOCK_D * BLOCK_H * BLOCK_W,
                    gap_acc + max_val,
                    gap_acc
                )
                valid_count += 1
    
    # Global average pooling (average over all maxpool outputs)
    if valid_count > 0:
        avg_val = tl.sum(gap_acc) / valid_count
    else:
        avg_val = 0.0
    
    # Clamp between 0 and 1
    avg_val = tl.minimum(tl.maximum(avg_val, 0.0), 1.0)
    
    # Store result [N, C, 1, 1, 1]
    out_idx = pid_n * stride_on + pid_c * stride_oc
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
    
    # Choose block sizes (powers of 2)
    BLOCK_C = min(triton.next_power_of_2(C), 32)
    BLOCK_D = 1  # We process one spatial block at a time
    BLOCK_H = 1
    BLOCK_W = 1
    
    # Grid configuration
    grid = (
        triton.cdiv(N, 1),
        triton.cdiv(C, BLOCK_C),
    )
    
    # Launch kernel
    fused_post_ops_kernel[grid](
        x, out,
        N, C, D, H, W,
        scale, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
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
