import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_tanh_scale_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Tanh: (exp(2*x) - 1) / (exp(2*x) + 1)
    exp_2x = tl.exp(2.0 * x)
    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Scale
    scaled = tanh_x * scaling_factor
    
    # Add bias (broadcast)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    result = scaled + bias
    
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    batch, channels, in_h, in_w, out_h, out_w,
    kernel_size, stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate output position
    total_out = batch * channels * out_h * out_w
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_out
    
    # Decompose linear index to (b, c, oh, ow)
    ow = idx % out_w
    temp = idx // out_w
    oh = temp % out_h
    temp = temp // out_h
    c = temp % channels
    b = temp // channels
    
    # Calculate input window start
    ih_start = oh * stride
    iw_start = ow * stride
    
    # Initialize max value as a vector (same shape as idx)
    max_val = tl.full([BLOCK_SIZE], -1e10, dtype=tl.float32)
    
    # Max pool over kernel window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = ih_start + kh
            iw = iw_start + kw
            
            # Check bounds
            valid = (ih < in_h) & (iw < in_w) & mask
            
            # Calculate input index
            in_idx = b * (channels * in_h * in_w) + c * (in_h * in_w) + ih * in_w + iw
            
            # Load value
            val = tl.load(x_ptr + in_idx, mask=valid, other=-1e10)
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + idx, max_val, mask=mask)


def fused_tanh_scale_bias(x, bias, scaling_factor):
    N = x.numel()
    out = torch.empty_like(x)
    
    # Flatten for processing
    x_flat = x.flatten()
    out_flat = out.flatten()
    
    # Broadcast bias to match x shape
    bias_expanded = bias.expand_as(x).flatten()
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    fused_tanh_scale_bias_kernel[grid](
        x_flat, bias_expanded, out_flat,
        N, scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def maxpool2d_triton(x, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    
    batch, channels, in_h, in_w = x.shape
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1
    
    out = torch.empty((batch, channels, out_h, out_w), device=x.device, dtype=x.dtype)
    
    total_out = batch * channels * out_h * out_w
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_out, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x, out,
        batch, channels, in_h, in_w, out_h, out_w,
        kernel_size, stride,
        BLOCK_SIZE=BLOCK_SIZE,
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
        # Convolution (keep PyTorch for correctness)
        x = self.conv(x)
        
        # Fused: Tanh + Scaling + Bias addition
        x = fused_tanh_scale_bias(x, self.bias, self.scaling_factor)
        
        # Max-pooling
        x = maxpool2d_triton(x, self.pool_kernel_size)
        
        return x
