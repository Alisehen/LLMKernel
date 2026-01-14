import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_tanh_scale_bias_maxpool_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    scaling_factor,
    pool_kernel_size,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Each program handles one output spatial location (h_out, w_out) for a batch of (n, c) pairs
    pid_spatial = tl.program_id(0)
    pid_batch_channel = tl.program_id(1)
    
    # Decode spatial position
    h_out = pid_spatial // W_out
    w_out = pid_spatial % W_out
    
    # Decode batch and channel
    n = pid_batch_channel // C
    c = pid_batch_channel % C
    
    # Early exit if out of bounds
    if h_out >= H_out or w_out >= W_out or n >= N or c >= C:
        return
    
    # Load bias for this channel
    bias_val = tl.load(bias_ptr + c)
    
    # Compute input window starting position
    h_in_start = h_out * pool_kernel_size
    w_in_start = w_out * pool_kernel_size
    
    # Initialize max value
    max_val = float('-inf')
    
    # Process pooling window
    for kh in range(pool_kernel_size):
        for kw in range(pool_kernel_size):
            h_in = h_in_start + kh
            w_in = w_in_start + kw
            
            # Check bounds
            if h_in < H_in and w_in < W_in:
                # Compute input offset
                in_offset = (n * stride_xn + c * stride_xc + 
                           h_in * stride_xh + w_in * stride_xw)
                
                # Load input value
                x_val = tl.load(x_ptr + in_offset)
                
                # Apply tanh: (exp(2*x) - 1) / (exp(2*x) + 1)
                exp_2x = tl.exp(2.0 * x_val)
                x_val = (exp_2x - 1.0) / (exp_2x + 1.0)
                
                # Apply scaling
                x_val = x_val * scaling_factor
                
                # Add bias
                x_val = x_val + bias_val
                
                # Update max
                max_val = tl.maximum(max_val, x_val)
    
    # Compute output offset
    out_offset = (n * stride_on + c * stride_oc + 
                 h_out * stride_oh + w_out * stride_ow)
    
    # Store result
    tl.store(out_ptr + out_offset, max_val)

def fused_tanh_scale_bias_maxpool(x, bias, scaling_factor, pool_kernel_size):
    N, C, H_in, W_in = x.shape
    H_out = H_in // pool_kernel_size
    W_out = W_in // pool_kernel_size
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    bias_flat = bias.view(-1)
    
    # Grid: (H_out * W_out, N * C)
    # Each program handles one output spatial location for one (n, c) pair
    grid = (H_out * W_out, N * C)
    
    fused_tanh_scale_bias_maxpool_kernel[grid](
        x, bias_flat, out,
        N, C, H_in, W_in, H_out, W_out,
        scaling_factor,
        pool_kernel_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_H=16,
        BLOCK_W=16,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Convolution (using PyTorch as it's highly optimized)
        x = self.conv(x)
        
        # Fused: Tanh + Scaling + Bias addition + Max-pooling
        x = fused_tanh_scale_bias_maxpool(x, self.bias, self.scaling_factor, self.pool_kernel_size)
        
        return x
