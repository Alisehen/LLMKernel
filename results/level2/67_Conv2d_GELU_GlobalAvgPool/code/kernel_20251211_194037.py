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
    
    # GELU approximation: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using tanh(z) = (e^{2z} - 1)/(e^{2z} + 1)
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    z = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    
    # Compute tanh using exp
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
):
    pid_b = tl.program_id(axis=0)
    pid_c_block = tl.program_id(axis=1)
    
    # Each block processes BLOCK_SIZE_C channels
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Initialize accumulator for each channel
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Process spatial dimensions in blocks
    for hw_start in range(0, H * W, BLOCK_SIZE_HW):
        hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < (H * W - hw_start)
        
        # Calculate full indices
        full_offsets = (pid_b * C * H * W + 
                       tl.reshape(c_offsets, (BLOCK_SIZE_C, 1)) * H * W +
                       hw_start + tl.reshape(hw_offsets, (1, BLOCK_SIZE_HW)))
        
        # Load and accumulate
        x_block = tl.load(x_ptr + full_offsets, mask=tl.reshape(c_mask, (BLOCK_SIZE_C, 1)) & tl.reshape(hw_mask, (1, BLOCK_SIZE_HW)))
        acc += tl.sum(x_block, axis=1)
    
    # Average and store
    output = acc / (H * W)
    output_offset = pid_b * C + c_offsets
    tl.store(output_ptr + output_offset, output, mask=c_mask)

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
):
    pid_b = tl.program_id(axis=0)
    pid_c_block = tl.program_id(axis=1)
    
    # Each block processes BLOCK_SIZE_C channels
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Initialize accumulator for each channel
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Constants for GELU
    sqrt_2_over_pi = 0.7978845608028654
    gelu_const = 0.044715
    
    # Process spatial dimensions in blocks
    for hw_start in range(0, H * W, BLOCK_SIZE_HW):
        hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
        hw_mask = hw_offsets < (H * W - hw_start)
        
        # Calculate full indices
        full_offsets = (pid_b * C * H * W + 
                       tl.reshape(c_offsets, (BLOCK_SIZE_C, 1)) * H * W +
                       hw_start + tl.reshape(hw_offsets, (1, BLOCK_SIZE_HW)))
        
        # Load
        x_block = tl.load(x_ptr + full_offsets, mask=tl.reshape(c_mask, (BLOCK_SIZE_C, 1)) & tl.reshape(hw_mask, (1, BLOCK_SIZE_HW)))
        
        # Apply GELU
        x_cubed = x_block * x_block * x_block
        z = sqrt_2_over_pi * (x_block + gelu_const * x_cubed)
        exp_2z = tl.exp(2.0 * z)
        tanh_z = (exp_2z - 1.0) / (exp_2z + 1.0)
        gelu_block = 0.5 * x_block * (1.0 + tanh_z)
        
        # Accumulate
        acc += tl.sum(gelu_block, axis=1)
    
    # Average and store
    output = acc / (H * W)
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
    
    # Optimize block sizes based on hardware constraints
    BLOCK_SIZE_C = 64  # Process 64 channels at once
    BLOCK_SIZE_HW = 256  # Process 256 spatial positions at once
    
    grid = (B, triton.cdiv(C, BLOCK_SIZE_C))
    global_avg_pool_kernel[grid](x, output, B, C, H, W, 
                                 BLOCK_SIZE_C=BLOCK_SIZE_C, 
                                 BLOCK_SIZE_HW=BLOCK_SIZE_HW)
    return output

def triton_fused_gelu_pool(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    output = torch.empty(B, C, device=x.device, dtype=x.dtype)
    
    # Optimize block sizes - tuned for A100
    BLOCK_SIZE_C = 64  # Good balance for register usage and parallelism
    BLOCK_SIZE_HW = 256  # Maximize spatial parallelism
    
    grid = (B, triton.cdiv(C, BLOCK_SIZE_C))
    fused_gelu_pool_kernel[grid](x, output, B, C, H, W,
                                BLOCK_SIZE_C=BLOCK_SIZE_C,
                                BLOCK_SIZE_HW=BLOCK_SIZE_HW)
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
