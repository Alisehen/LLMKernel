import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_residual_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    """
    Fused kernel for: BiasAdd + ResidualAdd + Multiply + ResidualAdd
    Input: [N, C, D, H, W], Output: [N, C, D, H, W]
    Performs: out = ((x + bias) + x) * x + x = 2*x^2 + bias*x + x
    Uses separate input and output buffers to avoid in-place corruption.
    """
    pid_n = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index
    
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_dhw_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    # Load bias for this channel (broadcast over spatial dimensions)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Process spatial blocks
    for block_idx in range(0, num_dhw_blocks):
        dhw_start = block_idx * BLOCK_DHW
        dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
        mask = dhw_offsets < DHW
        
        # Compute spatial indices
        d_offsets = dhw_offsets // (H * W)
        hw_offsets = dhw_offsets % (H * W)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W
        
        # Load input x (READ-ONLY)
        x_ptrs = (
            x_ptr + 
            pid_n * stride_xn + 
            pid_c * stride_xc + 
            d_offsets * stride_xd + 
            h_offsets * stride_xh + 
            w_offsets * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute fused operations: ((x + bias) + x) * x + x
        # Optimized as: 2*x^2 + bias*x + x
        x_squared = x_vals * x_vals
        result = x_squared * 2.0 + x_vals * bias_val + x_vals
        
        # Store result to separate output buffer
        out_ptrs = (
            out_ptr + 
            pid_n * stride_on + 
            pid_c * stride_oc + 
            d_offsets * stride_od + 
            h_offsets * stride_oh + 
            w_offsets * stride_ow
        )
        tl.store(out_ptrs, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_DHW': 512}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_DHW': 1024}, num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_DHW': 2048}, num_warps=8),
        triton.Config({'BLOCK_C': 16, 'BLOCK_DHW': 4096}, num_warps=8),
    ],
    key=['C', 'D', 'H', 'W']
)
@triton.jit
def fused_residual_kernel_autotune(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    """Same kernel with autotune configurations"""
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    if pid_n >= N or pid_c >= C:
        return
    
    DHW = D * H * W
    num_dhw_blocks = tl.cdiv(DHW, BLOCK_DHW)
    
    bias_val = tl.load(bias_ptr + pid_c)
    
    for block_idx in range(0, num_dhw_blocks):
        dhw_start = block_idx * BLOCK_DHW
        dhw_offsets = dhw_start + tl.arange(0, BLOCK_DHW)
        mask = dhw_offsets < DHW
        
        # Spatial indices
        d_offsets = dhw_offsets // (H * W)
        hw_offsets = dhw_offsets % (H * W)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W
        
        # Load input (READ-ONLY)
        x_ptrs = (
            x_ptr + 
            pid_n * stride_xn + 
            pid_c * stride_xc + 
            d_offsets * stride_xd + 
            h_offsets * stride_xh + 
            w_offsets * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Fused computation: 2*x^2 + bias*x + x
        x_squared = x_vals * x_vals
        result = x_squared * 2.0 + x_vals * bias_val + x_vals
        
        # Store to separate output buffer
        out_ptrs = (
            out_ptr + 
            pid_n * stride_on + 
            pid_c * stride_oc + 
            d_offsets * stride_od + 
            h_offsets * stride_oh + 
            w_offsets * stride_ow
        )
        tl.store(out_ptrs, result, mask=mask)


def fused_post_convtranspose(x, bias):
    """
    Fused operations: BiasAdd + ResidualAdd + Multiply + ResidualAdd
    Uses separate output buffer to avoid in-place corruption.
    """
    N, C, D, H, W = x.shape
    
    # Ensure bias has correct shape for broadcasting
    bias = bias.view(-1, 1, 1, 1)  # [C, 1, 1, 1]
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate grid
    grid = (N, C)
    
    # Call kernel with autotune
    fused_residual_kernel_autotune[grid](
        x, bias, out,  # separate input and output
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused post-ops (Triton)
    
    Note: ConvTranspose3d is kept in PyTorch due to complexity.
    Only the subsequent operations are fused in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Bias parameter (broadcastable to [out_channels, 1, 1, 1])
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
    
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused post-ops in Triton (with separate output)
        x = fused_post_convtranspose(x, self.bias)
        
        return x
