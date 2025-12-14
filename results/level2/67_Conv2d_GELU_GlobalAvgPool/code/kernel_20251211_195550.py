import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Fast GELU approximation with reduced operations
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = x + 0.044715 * x_cubed
    z = sqrt_2_over_pi * inner
    exp_2z = tl.exp(2.0 * z)
    tanh_z = (exp_2z - 1.0) / (exp_2z + 1.0)
    
    output = 0.5 * x * (1.0 + tanh_z)
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c_block = tl.program_id(axis=1)
    
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Initialize accumulator with float32 for stable accumulation
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    hw_size = H * W
    
    # Process spatial dimensions in blocks
    for hw_start in range(0, hw_size, BLOCK_SIZE_HW):
        hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < (hw_size - hw_start)
        
        # Optimized address calculation
        base_offset = pid_b * C * hw_size
        channel_offset = tl.reshape(c_offsets, (BLOCK_SIZE_C, 1)) * hw_size
        spatial_offset = hw_start + tl.reshape(hw_offsets, (1, BLOCK_SIZE_HW))
        
        full_offsets = base_offset + channel_offset + spatial_offset
        
        # Load without cache modifier
        x_block = tl.load(x_ptr + full_offsets,
                         mask=tl.reshape(c_mask, (BLOCK_SIZE_C, 1)) & 
                              tl.reshape(hw_mask, (1, BLOCK_SIZE_HW)))
        
        # Accumulate
        acc += tl.sum(x_block.to(tl.float32), axis=1)
    
    # Average and store
    output = acc / hw_size
    output_offset = pid_b * C + c_offsets
    tl.store(output_ptr + output_offset, output, mask=c_mask)

# Enhanced autotune configurations
configs = [
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 64, 'num_stages': 1}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 64, 'num_stages': 2}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_HW': 64, 'num_stages': 3}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_HW': 32, 'num_stages': 1}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_HW': 32, 'num_stages': 2}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 128, 'num_stages': 1}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 128, 'num_stages': 2}, num_warps=8),
    triton.Config({'BLOCK_SIZE_C': 96, 'BLOCK_SIZE_HW': 48, 'num_stages': 1}, num_warps=8),
]

@triton.autotune(
    configs=configs,
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_gelu_pool_kernel(
    x_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c_block = tl.program_id(axis=1)
    
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Initialize accumulator with float32 precision
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    hw_size = H * W
    
    # Constants for GELU
    sqrt_2_over_pi = 0.7978845608028654
    gelu_const = 0.044715
    
    # Process spatial dimensions in blocks
    for hw_start in range(0, hw_size, BLOCK_SIZE_HW):
        hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < (hw_size - hw_start)
        
        # Optimized address calculation
        base_offset = pid_b * C * hw_size
        channel_offset = tl.reshape(c_offsets, (BLOCK_SIZE_C, 1)) * hw_size
        spatial_offset = hw_start + tl.reshape(hw_offsets, (1, BLOCK_SIZE_HW))
        
        full_offsets = base_offset + channel_offset + spatial_offset
        
        # Load without cache modifier
        x_block = tl.load(x_ptr + full_offsets,
                         mask=tl.reshape(c_mask, (BLOCK_SIZE_C, 1)) & 
                              tl.reshape(hw_mask, (1, BLOCK_SIZE_HW)))
        
        # Apply GELU with optimized computation
        x_block_f32 = x_block.to(tl.float32)
        x_cubed = x_block_f32 * x_block_f32 * x_block_f32
        inner = x_block_f32 + gelu_const * x_cubed
        z = sqrt_2_over_pi * inner
        
        # Stable tanh computation
        exp_2z = tl.exp(2.0 * z)
        tanh_z = 1.0 - 2.0 / (exp_2z + 1.0)
        gelu_block = 0.5 * x_block_f32 * (1.0 + tanh_z)
        
        # Accumulate in float32
        acc += tl.sum(gelu_block, axis=1)
    
    # Average and store
    output = acc / hw_size
    output_offset = pid_b * C + c_offsets
    tl.store(output_ptr + output_offset, output, mask=c_mask)

def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    output = torch.empty(B, C, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_HW = 64
    num_stages = 2
    
    grid = (B, triton.cdiv(C, BLOCK_SIZE_C))
    global_avg_pool_kernel[grid](x, output, B, C, H, W, 
                                 BLOCK_SIZE_C=BLOCK_SIZE_C, 
                                 BLOCK_SIZE_HW=BLOCK_SIZE_HW,
                                 num_stages=num_stages)
    return output

def triton_fused_gelu_pool(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    output = torch.empty(B, C, device=x.device, dtype=x.dtype)
    
    # Launch autotuned kernel
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_SIZE_C']))
    fused_gelu_pool_kernel[grid](x, output, B, C, H, W)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # Keep PyTorch convolution (highly optimized with cuDNN)
        x = self.conv(x)
        
        # Fused GELU + Global Average Pooling for maximum performance
        x = triton_fused_gelu_pool(x)
        
        return x
