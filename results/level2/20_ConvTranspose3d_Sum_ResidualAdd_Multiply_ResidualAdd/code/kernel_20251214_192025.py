import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative configs for register pressure (avoid spilling)
        triton.Config({'BLOCK_DHW': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_DHW': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_DHW': 1024}, num_warps=8, num_stages=2),
        # More aggressive for larger spatial dimensions
        triton.Config({'BLOCK_DHW': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_DHW': 4096}, num_warps=16, num_stages=3),
    ],
    key=['D', 'H', 'W', 'C']
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
    Ultra-optimized kernel with register pressure optimization.
    Key improvements:
    1. Reduced register usage by recomputing cheap ops instead of storing
    2. Minimized intermediate variables
    3. Optimized memory access patterns for Ada Lovelace
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    
    # Early exit for boundary cases
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_spatial_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    if pid_dhw >= num_spatial_blocks:
        return
    
    # Load bias once per channel (reuse across all spatial elements)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Calculate spatial offsets with minimal register usage
    dhw_start = pid_dhw * BLOCK_DHW
    dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
    mask = dhw_offsets < DHW
    
    # Compute 3D indices directly without intermediate storage
    HW = H * W
    
    # Base pointers computed once
    x_base = x_ptr + pid_n * stride_xn + pid_c * stride_xc
    out_base = out_ptr + pid_n * stride_on + pid_c * stride_oc
    
    # Process spatial elements with minimal register footprint
    for idx in tl.static_range(0, BLOCK_DHW):
        dhw_offset = dhw_start + idx
        if tl.multiple_of(dhw_offset < DHW, 8):  # Hint for compiler optimization
            
            # Compute indices without storing intermediates
            d_idx = dhw_offset // HW
            hw_rem = dhw_offset % HW
            h_idx = hw_rem // W
            w_idx = hw_rem % W
            
            # Load and compute with minimal intermediates
            x_val = tl.load(
                x_base + d_idx * stride_xd + h_idx * stride_xh + w_idx * stride_xw
            )
            
            # Fused computation: 2*x^2 + bias*x + x
            # Recompute x^2 instead of storing if register pressure is high
            # TLDR: Keep x^2 (moderate cost) but compute result directly
            result = x_val * (bias_val + 1.0) + x_val * x_val * 2.0
            
            tl.store(
                out_base + d_idx * stride_od + h_idx * stride_oh + w_idx * stride_ow,
                result
            )


def fused_post_convtranspose_optimized(x, bias):
    """
    Optimized wrapper with adaptive grid calculation for Ada Lovelace.
    """
    N, C, D, H, W = x.shape
    DHW = D * H * W
    
    # Prepare output
    out = torch.empty_like(x)
    
    # Calculate optimal grid size for 4090 (128 SMs)
    # Target: 4-8 blocks per SM for occupancy
    max_threads_per_sm = 1536
    num_sms = 128
    
    # Conservative spatial block count to avoid register spilling
    # Use smaller blocks when channels are large (more register pressure)
    spatial_blocks = triton.cdiv(DHW, 512)  # Start with safe default
    
    grid = (N, C, spatial_blocks)
    
    # Launch kernel
    fused_residual_kernel_optimized[grid](
        x, bias, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized ModelNew with register-aware kernel fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Bias shape matches channel dimension for broadcasting
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
    
    def forward(self, x):
        # ConvTranspose operation
        x = self.conv_transpose(x)
        
        # Fused post-processing with optimized kernel
        x = fused_post_convtranspose_optimized(x, self.bias)
        
        return x
