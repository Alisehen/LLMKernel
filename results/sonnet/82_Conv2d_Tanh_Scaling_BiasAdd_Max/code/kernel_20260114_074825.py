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
def maxpool2d_tiled_kernel(
    x_ptr, out_ptr,
    batch, channels, in_h, in_w, out_h, out_w,
    kernel_size, stride,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # 2D grid: (batch * channels, out_h * out_w / BLOCK)
    pid_bc = tl.program_id(0)  # batch * channels dimension
    pid_spatial = tl.program_id(1)  # spatial dimension
    
    # Decompose pid_bc into batch and channel
    c = pid_bc % channels
    b = pid_bc // channels
    
    # Calculate spatial output positions for this block
    # Each block handles BLOCK_H x BLOCK_W output positions
    block_oh = (pid_spatial // ((out_w + BLOCK_W - 1) // BLOCK_W)) * BLOCK_H
    block_ow = (pid_spatial % ((out_w + BLOCK_W - 1) // BLOCK_W)) * BLOCK_W
    
    # Thread offsets within block
    oh_offs = block_oh + tl.arange(0, BLOCK_H)
    ow_offs = block_ow + tl.arange(0, BLOCK_W)
    
    # Create 2D mask for output positions
    oh_mask = oh_offs < out_h
    ow_mask = ow_offs < out_w
    
    # Compute input window start positions
    ih_start = oh_offs * stride
    iw_start = ow_offs * stride
    
    # Initialize max values for each output position
    max_vals = tl.full([BLOCK_H, BLOCK_W], -1e10, dtype=tl.float32)
    
    # Base input pointer for this batch and channel
    base_in_ptr = b * (channels * in_h * in_w) + c * (in_h * in_w)
    
    # Iterate over kernel window
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Input positions for this kernel element
            ih = ih_start[:, None] + kh
            iw = iw_start[None, :] + kw
            
            # Bounds check
            valid = (ih < in_h) & (iw < in_w) & oh_mask[:, None] & ow_mask[None, :]
            
            # Calculate input indices
            in_idx = base_in_ptr + ih * in_w + iw
            
            # Load values with coalesced access pattern
            vals = tl.load(x_ptr + in_idx, mask=valid, other=-1e10)
            
            # Update max
            max_vals = tl.maximum(max_vals, vals)
    
    # Calculate output indices
    out_idx = b * (channels * out_h * out_w) + c * (out_h * out_w) + oh_offs[:, None] * out_w + ow_offs[None, :]
    
    # Store results
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_idx, max_vals, mask=out_mask)


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
    
    # Use 2D tiled approach
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid_bc = batch * channels
    grid_h = triton.cdiv(out_h, BLOCK_H)
    grid_w = triton.cdiv(out_w, BLOCK_W)
    grid_spatial = grid_h * grid_w
    
    grid = (grid_bc, grid_spatial)
    
    maxpool2d_tiled_kernel[grid](
        x, out,
        batch, channels, in_h, in_w, out_h, out_w,
        kernel_size, stride,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
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
        
        # Max-pooling with optimized tiled kernel
        x = maxpool2d_triton(x, self.pool_kernel_size)
        
        return x
