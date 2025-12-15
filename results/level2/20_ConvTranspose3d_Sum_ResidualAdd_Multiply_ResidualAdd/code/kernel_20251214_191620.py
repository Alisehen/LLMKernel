import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_residual_kernel_optimized(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_DHW: tl.constexpr,
):
    """
    Optimized fused kernel with 3D grid layout.
    Grid: (N, C, spatial_blocks) - maximizes parallelism
    All operations use identical offset calculations and masks
    """
    pid_n = tl.program_id(0)  # batch dimension
    pid_c = tl.program_id(1)  # channel dimension
    pid_dhw = tl.program_id(2)  # spatial block dimension
    
    # Boundary checks
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_spatial_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    if pid_dhw >= num_spatial_blocks:
        return
    
    # Load bias once per channel (broadcast)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Calculate spatial offsets - shared by ALL operations
    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
    mask = dhw_offsets < DHW
    
    # Convert linear spatial index to 3D indices
    # Use fast integer division/modulo operations
    HW = H * W
    d_offsets = dhw_offsets // HW
    hw_offsets = dhw_offsets % HW
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W
    
    # Base pointer calculations - computed ONCE, used for ALL loads/stores
    x_base = (
        x_ptr + 
        pid_n * stride_xn + 
        pid_c * stride_xc
    )
    
    # Load input values using shared offsets and mask
    x_vals = tl.load(
        x_base + 
        d_offsets * stride_xd + 
        h_offsets * stride_xh + 
        w_offsets * stride_xw,
        mask=mask,
        other=0.0
    )
    
    # Fused computation: 2*x^2 + bias*x + x
    # Optimized for FMA (fused multiply-add) instructions
    x_squared = x_vals * x_vals
    result = tl.fma(x_vals, bias_val, x_squared * 2.0 + x_vals)
    
    # Store using same offsets and mask
    out_base = (
        out_ptr + 
        pid_n * stride_on + 
        pid_c * stride_oc
    )
    
    tl.store(
        out_base + 
        d_offsets * stride_od + 
        h_offsets * stride_oh + 
        w_offsets * stride_ow,
        result,
        mask=mask
    )


@triton.autotune(
    configs=[
        # Maximize occupancy: 1024 threads, 32 warps (fully utilizes SM)
        triton.Config({'BLOCK_DHW': 1024}, num_warps=32, num_stages=3),
        # Balance memory latency hiding
        triton.Config({'BLOCK_DHW': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_DHW': 4096}, num_warps=8, num_stages=5),
        # For smaller spatial dimensions
        triton.Config({'BLOCK_DHW': 512}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_DHW': 256}, num_warps=8, num_stages=4),
    ],
    key=['D', 'H', 'W', 'C']
)
@triton.jit
def fused_residual_kernel_autotune_optimized(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_DHW: tl.constexpr,
):
    """Autotuned version with optimized configurations"""
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_spatial_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    if pid_dhw >= num_spatial_blocks:
        return
    
    bias_val = tl.load(bias_ptr + pid_c)
    
    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
    mask = dhw_offsets < DHW
    
    HW = H * W
    d_offsets = dhw_offsets // HW
    hw_offsets = dhw_offsets % HW
    h_offsets = hw_offsets // W
    w_offsets = hw_offsets % W
    
    x_base = (
        x_ptr + 
        pid_n * stride_xn + 
        pid_c * stride_xc
    )
    
    x_vals = tl.load(
        x_base + 
        d_offsets * stride_xd + 
        h_offsets * stride_xh + 
        w_offsets * stride_xw,
        mask=mask,
        other=0.0
    )
    
    x_squared = x_vals * x_vals
    result = tl.fma(x_vals, bias_val, x_squared * 2.0 + x_vals)
    
    out_base = (
        out_ptr + 
        pid_n * stride_on + 
        pid_c * stride_oc
    )
    
    tl.store(
        out_base + 
        d_offsets * stride_od + 
        h_offsets * stride_oh + 
        w_offsets * stride_ow,
        result,
        mask=mask
    )


def fused_post_convtranspose_optimized(x, bias):
    """
    Optimized wrapper with 3D grid layout for maximum parallelism.
    """
    N, C, D, H, W = x.shape
    
    # Prepare bias for broadcasting
    bias = bias.view(-1, 1, 1, 1)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate spatial blocks
    DHW = D * H * W
    
    # Use 3D grid: (N, C, spatial_blocks) - maximizes GPU utilization
    # Each block processes BLOCK_DHW spatial elements
    spatial_blocks = triton.cdiv(DHW, 1024)  # Start with max threads for grid calc
    
    grid = (N, C, spatial_blocks)
    
    # Launch optimized kernel
    fused_residual_kernel_autotune_optimized[grid](
        x, bias, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized ModelNew with 3D grid layout for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
    
    def forward(self, x):
        # PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Optimized fused post-ops with 3D grid
        x = fused_post_convtranspose_optimized(x, self.bias)
        
        return x
