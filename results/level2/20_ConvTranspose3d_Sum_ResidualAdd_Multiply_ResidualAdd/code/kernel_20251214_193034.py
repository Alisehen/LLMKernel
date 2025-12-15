import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Conservative baseline - minimal register pressure
        triton.Config({'BLOCK_DHW': 1024}, num_warps=4, num_stages=2),
        # Balanced for compute-bound operations
        triton.Config({'BLOCK_DHW': 2048}, num_warps=8, num_stages=3),
        # Memory-bound optimization with reduced warps
        triton.Config({'BLOCK_DHW': 512}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W']
)
@triton.jit
def fused_residual_kernel_optimized(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_DHW: tl.constexpr,
):
    """
    Optimized fused kernel with minimal register pressure and optimized memory access.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    
    # Early exit for out-of-bounds
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_spatial_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    if pid_dhw >= num_spatial_blocks:
        return
    
    # Load bias once - broadcast across spatial dimensions
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Calculate spatial offsets with minimal temporary registers
    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
    mask = dhw_offsets < DHW
    
    # Precompute HW once to avoid repeated computation
    HW = H * W
    # Compute 3D indices using fused operations
    d_idx = dhw_offsets // HW
    hw_remainder = dhw_offsets - d_idx * HW  # Avoid modulo
    h_idx = hw_remainder // W
    w_idx = hw_remainder - h_idx * W  # Avoid modulo
    
    # Single base pointer calculation for input
    x_base = x_ptr + pid_n * stride_xn + pid_c * stride_xc
    
    # Coalesced memory access with single offset calculation
    x_vals = tl.load(
        x_base + d_idx * stride_xd + h_idx * stride_xh + w_idx * stride_xw,
        mask=mask,
        other=0.0
    )
    
    # Fused computation: 2*x^2 + bias*x + x
    # Reuse x_vals to minimize registers
    result = tl.fma(x_vals, bias_val, x_vals * (x_vals * 2.0 + 1.0))
    
    # Single base pointer calculation for output
    out_base = out_ptr + pid_n * stride_on + pid_c * stride_oc
    
    # Coalesced store with same indexing
    tl.store(
        out_base + d_idx * stride_od + h_idx * stride_oh + w_idx * stride_ow,
        result,
        mask=mask
    )


def fused_post_convtranspose_optimized(x, bias):
    """
    Optimized wrapper with adaptive grid calculation for maximum SM utilization.
    """
    N, C, D, H, W = x.shape
    
    # Ensure bias is correctly shaped for broadcasting
    bias = bias.view(-1, 1, 1, 1)
    
    # Pre-allocate output
    out = torch.empty_like(x)
    
    # Calculate total spatial elements
    DHW = D * H * W
    
    # Use minimum block size from autotune configs for grid calculation
    min_block_size = 512  # From autotune configs
    spatial_blocks = triton.cdiv(DHW, min_block_size)
    
    # Optimized grid for Ada Lovelace architecture
    grid = (N, C, spatial_blocks)
    
    # Launch optimized kernel
    fused_residual_kernel_optimized[grid](
        x, bias, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized ModelNew with register-pressure aware kernel configurations.
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
        
        # Optimized fused post-ops with register-aware configurations
        x = fused_post_convtranspose_optimized(x, self.bias)
        
        return x
