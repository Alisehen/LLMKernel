import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        # Optimized for Ada Lovelace 4090: maximize occupancy with vectorization
        triton.Config({'BLOCK_W': 256, 'VEC': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 128, 'VEC': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_W': 512, 'VEC': 2}, num_warps=16, num_stages=2),
        # Balance memory latency hiding with vectorization
        triton.Config({'BLOCK_W': 256, 'VEC': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_W': 128, 'VEC': 4}, num_warps=4, num_stages=4),
        # Maximum vectorization for contiguous W dimension
        triton.Config({'BLOCK_W': 64, 'VEC': 16}, num_warps=2, num_stages=5),
    ],
    key=['W', 'H', 'D', 'C']
)
@triton.jit
def fused_residual_kernel_vectorized(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_W: tl.constexpr,
    VEC: tl.constexpr,
):
    """
    Optimized kernel with vectorized loads/stores along W dimension.
    Maximizes memory coalescing by leveraging contiguous W dimension.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dh = tl.program_id(2)
    
    # Early exit for invalid program IDs
    if pid_n >= N or pid_c >= C:
        return
    
    # Decompose pid_dh into d and h
    DH = D * H
    if pid_dh >= DH:
        return
    d = pid_dh // H
    h = pid_dh % H
    
    # Load bias once per channel (broadcast)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Vectorized processing along W dimension
    w_offsets = tl.arange(0, BLOCK_W)
    w_mask = w_offsets < W
    
    # Base pointer calculations - computed ONCE
    x_base = (
        x_ptr + 
        pid_n * stride_xn + 
        pid_c * stride_xc + 
        d * stride_xd + 
        h * stride_xh
    )
    
    out_base = (
        out_ptr + 
        pid_n * stride_on + 
        pid_c * stride_oc + 
        d * stride_od + 
        h * stride_oh
    )
    
    # Vectorized load and compute loop
    for w_start in range(0, W, BLOCK_W * VEC):
        w_vec_offsets = w_start + w_offsets[:, None] * VEC + tl.arange(0, VEC)[None, :]
        vec_mask = w_vec_offsets < W
        
        # Vectorized load with proper masking
        x_vals_vec = tl.load(
            x_base + w_vec_offsets * stride_xw,
            mask=vec_mask,
            other=0.0
        )
        
        # Fused computation with vector operations
        x_squared_vec = x_vals_vec * x_vals_vec
        result_vec = tl.fma(x_vals_vec, bias_val, x_squared_vec * 2.0 + x_vals_vec)
        
        # Vectorized store
        tl.store(
            out_base + w_vec_offsets * stride_ow,
            result_vec,
            mask=vec_mask
        )

@triton.autotune(
    configs=[
        # Optimized for spatial dimensions with proper vectorization
        triton.Config({'BLOCK_SIZE': 1024, 'VEC': 4}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'VEC': 8}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 256, 'VEC': 16}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC': 2}, num_warps=32, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W']
)
@triton.jit
def fused_residual_kernel_3d_vectorized(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
    VEC: tl.constexpr,
):
    """
    Alternative kernel with 3D tiling and vectorization for better cache locality.
    Uses 3D block structure: (N, C, spatial_tiles) with vectorized spatial processing.
    """
    # 3D block indexing
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(N * C * D * H * W, BLOCK_SIZE)
    
    if pid >= num_tiles:
        return
    
    # Decompose tile into N, C, D, H, W indices
    tile_start = pid * BLOCK_SIZE
    tile_idx = tile_start + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear index to 5D indices
    n_idx = tile_idx // (C * D * H * W)
    remainder = tile_idx % (C * D * H * W)
    c_idx = remainder // (D * H * W)
    remainder = remainder % (D * H * W)
    d_idx = remainder // (H * W)
    remainder = remainder % (H * W)
    h_idx = remainder // W
    w_idx = remainder % W
    
    # Mask for valid indices
    mask = (n_idx < N) & (c_idx < C) & (d_idx < D) & (h_idx < H) & (w_idx < W)
    
    # Load bias with proper indexing
    bias_vals = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Calculate pointer offsets - vectorized for better memory throughput
    x_offsets = (
        n_idx * stride_xn + 
        c_idx * stride_xc + 
        d_idx * stride_xd + 
        h_idx * stride_xh + 
        w_idx * stride_xw
    )
    
    # Vectorized loads with proper masking
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    
    # Fused computation - all in registers
    x_squared = x_vals * x_vals
    result = tl.fma(x_vals, bias_vals, x_squared * 2.0 + x_vals)
    
    # Calculate output offsets
    out_offsets = (
        n_idx * stride_on + 
        c_idx * stride_oc + 
        d_idx * stride_od + 
        h_idx * stride_oh + 
        w_idx * stride_ow
    )
    
    # Single store for final result
    tl.store(out_ptr + out_offsets, result, mask=mask)

def fused_post_convtranspose_optimized(x, bias):
    """
    Optimized wrapper that selects the best kernel based on tensor shape.
    Uses vectorized kernel for W-contiguous data, 3D kernel for others.
    """
    N, C, D, H, W = x.shape
    
    # Prepare bias for broadcasting
    bias = bias.view(-1, 1, 1, 1)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Heuristic: Use vectorized kernel if W is power of 2 and >= 32
    # This ensures good vectorization and memory coalescing
    if W >= 32 and (W & (W - 1)) == 0:
        # Vectorized kernel along W dimension
        DH = D * H
        grid = (N, C, DH)
        
        fused_residual_kernel_vectorized[grid](
            x, bias, out,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        )
    else:
        # 3D vectorized kernel for general case
        total_elements = N * C * D * H * W
        grid_size = triton.cdiv(total_elements, 1024)  # Start with max threads
        
        fused_residual_kernel_3d_vectorized[grid_size](
            x, bias, out,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        )
    
    return out

class ModelNew(nn.Module):
    """
    Optimized ModelNew with fused operations using vectorized Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Ensure bias has correct shape for broadcasting
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1))
    
    def forward(self, x):
        # PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Optimized fused post-ops with vectorized kernels
        x = fused_post_convtranspose_optimized(x, self.bias)
        
        return x
