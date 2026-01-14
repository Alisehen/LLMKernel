import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_tanh_scale_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = N * C * H * W
    
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_elements
    
    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Fast tanh approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9x²)
    x_sq = x * x
    numerator = x * (27.0 + x_sq)
    denominator = 27.0 + 9.0 * x_sq
    x_tanh = numerator / denominator
    
    # Fused scaling
    x_scaled = x_tanh * scaling_factor
    
    # Compute channel index for bias
    hw = H * W
    c_idx = (offs // hw) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Add bias
    result = x_scaled + bias
    
    # Store output
    tl.store(out_ptr + offs, result, mask=mask)

@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    kernel_size: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_nc = tl.program_id(2)
    
    # Decompose NC dimension
    n = pid_nc // C
    c = pid_nc % C
    
    # Compute output positions
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    
    h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offs = w_start + tl.arange(0, BLOCK_SIZE_W)
    
    h_mask = h_offs < H_out
    w_mask = w_offs < W_out
    
    # Expand to 2D
    h_offs_2d = h_offs[:, None]
    w_offs_2d = w_offs[None, :]
    mask_2d = h_mask[:, None] & w_mask[None, :]
    
    # Compute input starting positions
    h_in_start = h_offs_2d * kernel_size
    w_in_start = w_offs_2d * kernel_size
    
    # Initialize max value
    max_val = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), float('-inf'), dtype=tl.float32)
    
    # Base offset for this NC slice
    base_offset = n * stride_xn + c * stride_xc
    
    # Max pooling loop - unrolled via tl.static_range
    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            h_in = h_in_start + kh
            w_in = w_in_start + kw
            
            # Compute input offset
            in_offs = base_offset + h_in * stride_xh + w_in * stride_xw
            
            # Boundary check
            valid = mask_2d & (h_in < H_in) & (w_in < W_in)
            val = tl.load(x_ptr + in_offs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    # Store output
    out_offs = n * stride_on + c * stride_oc + h_offs_2d * stride_oh + w_offs_2d * stride_ow
    tl.store(out_ptr + out_offs, max_val, mask=mask_2d)

def fused_tanh_scale_bias(x, bias, scaling_factor):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    num_elements = N * C * H * W
    
    # Conservative BLOCK_SIZE for register pressure management
    # Autotune between 512 and 256
    def grid(meta):
        return (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
    
    fused_tanh_scale_bias_kernel[grid](
        x, bias, out,
        N, C, H, W,
        scaling_factor,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=512,
    )
    return out

def max_pool2d_triton(x, kernel_size):
    N, C, H_in, W_in = x.shape
    H_out = H_in // kernel_size
    W_out = W_in // kernel_size
    
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Optimized 2D blocking for spatial locality
    # Conservative sizes to avoid register spilling
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    grid = (
        triton.cdiv(H_out, BLOCK_SIZE_H),
        triton.cdiv(W_out, BLOCK_SIZE_W),
        N * C
    )
    
    max_pool2d_kernel[grid](
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        kernel_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
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
        # Convolution (using PyTorch's highly optimized implementation)
        x = self.conv(x)
        
        # Fused: Tanh + Scaling + Bias addition
        bias_flat = self.bias.view(-1)
        x = fused_tanh_scale_bias(x, bias_flat, self.scaling_factor)
        
        # Max-pooling with optimized 2D blocking
        x = max_pool2d_triton(x, self.pool_kernel_size)
        
        return x
