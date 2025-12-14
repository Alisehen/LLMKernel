import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_activation_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    scaling_factor,
    n_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute channel index for bias broadcasting: (offsets // (H*W)) % C
    HW = H * W
    c_idx = (offsets // HW) % C
    bias = tl.load(bias_ptr + c_idx)
    
    # tanh using exponential form with scalar operations (no full_like)
    exp_2x = tl.exp(2.0 * x)
    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    activated = tanh_x * scaling_factor + bias
    tl.store(output_ptr + offsets, activated, mask=mask)


@triton.jit
def max_pool_2d_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    pool_h, pool_w,
    stride_h, stride_w,
    output_h, output_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    
    output_size = B * C * output_h * output_w
    positions_per_program = tl.cdiv(output_size, num_programs)
    start_pos = pid * positions_per_program
    end_pos = min(start_pos + positions_per_program, output_size)
    
    for pos in range(start_pos, end_pos, BLOCK_SIZE):
        offsets = pos + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_pos
        
        # Compute output indices
        w_idx = offsets % output_w
        h_idx = (offsets // output_w) % output_h
        c_idx = (offsets // (output_w * output_h)) % C
        b_idx = offsets // (output_w * output_h * C)
        
        # Input window indices
        h_start = h_idx * stride_h
        w_start = w_idx * stride_w
        h_end = min(h_start + pool_h, H)
        w_end = min(w_start + pool_w, W)
        
        # Initialize with minimum float value
        min_val = -3.4028235e38
        max_val = tl.full((BLOCK_SIZE,), min_val, dtype=tl.float32)
        
        # Pooling window
        for ph in range(pool_h):
            ph_actual = h_start + ph
            for pw in range(pool_w):
                pw_actual = w_start + pw
                
                # Calculate input index
                input_idx = (b_idx * C * H * W + 
                            c_idx * H * W + 
                            ph_actual * W + 
                            pw_actual)
                
                # Load with bounds checking
                h_in_bounds = (ph_actual < h_end) & (ph_actual >= 0)
                w_in_bounds = (pw_actual < w_end) & (pw_actual >= 0)
                load_mask = mask & h_in_bounds & w_in_bounds
                val = tl.load(input_ptr + input_idx, mask=load_mask, other=min_val)
                
                # Vectorized max
                max_val = tl.where(val > max_val, val, max_val)
        
        # Calculate output index and store
        output_idx = (b_idx * C * output_h * output_w + 
                     c_idx * output_h * output_w + 
                     h_idx * output_w + 
                     w_idx)
        tl.store(output_ptr + output_idx, max_val, mask=mask)


def triton_fused_activation(x: torch.Tensor, scaling_factor: float, bias: torch.Tensor) -> torch.Tensor:
    # Flatten bias to (C,) for proper broadcasting
    if bias.dim() > 1:
        bias = bias.view(-1)
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    B, C, H, W = x.shape
    
    # Optimize BLOCK_SIZE for elementwise operations
    BLOCK_SIZE = min(1024, triton.next_power_of_2(H * W))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_activation_kernel[grid](
        x, bias, output, scaling_factor, n_elements,
        C, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


def triton_max_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    stride = kernel_size
    pool_h = pool_w = kernel_size
    
    output_h = H // stride
    output_w = W // stride
    output = torch.empty((B, C, output_h, output_w), device=x.device, dtype=x.dtype)
    
    if output_h <= 0 or output_w <= 0:
        raise ValueError(f"Invalid output dimensions: {output_h}x{output_w}")
    
    # Use 1D grid with optimized block size
    BLOCK_SIZE = min(256, triton.next_power_of_2(output_h * output_w))
    output_size = B * C * output_h * output_w
    grid_size = min(65535, triton.cdiv(output_size, BLOCK_SIZE))
    grid = (max(1, grid_size),)
    
    max_pool_2d_kernel[grid](
        x, output,
        B, C, H, W,
        pool_h, pool_w,
        stride, stride,
        output_h, output_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        
    def forward(self, x):
        # Use PyTorch convolution for stability
        x = self.conv(x)
        
        # Fused activation (tanh + scaling + bias)
        x = triton_fused_activation(x, self.scaling_factor, self.bias)
        
        # Max pooling
        x = triton_max_pool2d(x, self.pool_kernel_size)
        
        return x
