import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def swish_activation_fast(x):
    """Optimized Swish activation using fast sigmoid approximation"""
    # Fast sigmoid: 1 / (1 + exp(-x)) ≈ (tanh(x/2) + 1) / 2
    # Use piecewise approximation for better performance
    half_x = x * 0.5
    # tanh approximation using rational form
    x_sq = half_x * half_x
    tanh_approx = half_x * (27.0 + x_sq) / (27.0 + 9.0 * x_sq)
    sigmoid = (tanh_approx + 1.0) * 0.5
    return x * sigmoid


@triton.jit
def hardswish_activation_fast(x):
    """Optimized HardSwish with fewer branches"""
    # Compute relu6(x + 3) using min/max
    x_plus_3 = x + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    return x * relu6 * (1.0 / 6.0)  # Multiply by reciprocal


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_SIZE': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W', 'groups'],
)
@triton.jit
def fused_swish_groupnorm_hardswish_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N, C, D, H, W,
    groups,
    eps,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel: Swish + GroupNorm + HardSwish
    - Single pass with Welford's algorithm for online mean/variance
    - Precomputed spatial indices
    - Better memory access patterns
    """
    pid_n = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index
    
    if pid_n >= N or pid_g >= groups:
        return
    
    # Calculate group parameters
    group_size = C // groups
    start_c = pid_g * group_size
    
    # Precompute spatial indices once
    DHW = D * H * W
    offs_spatial = tl.arange(0, BLOCK_SIZE)
    spatial_mask = offs_spatial < DHW
    
    # Precompute spatial indices for the entire block
    d_idx = tl.where(spatial_mask, offs_spatial // (H * W), 0)
    h_idx = tl.where(spatial_mask, (offs_spatial % (H * W)) // W, 0)
    w_idx = tl.where(spatial_mask, offs_spatial % W, 0)
    
    # Welford's algorithm for online mean/variance computation
    channel_offsets = tl.arange(0, BLOCK_C)
    channel_mask = channel_offsets < group_size
    
    # Initialize accumulators
    mean = tl.zeros((BLOCK_C,), dtype=tl.float32)
    m2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count = 0.0
    
    # Process spatial blocks
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        spatial_valid = spatial_mask & (offs_spatial >= spatial_base) & (offs_spatial < spatial_base + BLOCK_SIZE)
        
        if not tl.sum(spatial_valid):
            continue
            
        # Process channels in vectorized manner
        for c_base in range(0, group_size, BLOCK_C):
            c_idx = start_c + c_base + channel_offsets
            c_valid = channel_mask & ((c_base + channel_offsets) < group_size)
            
            if not tl.sum(c_valid):
                continue
                
            # Load input with optimized addressing
            x_ptrs = (x_ptr + 
                     pid_n * stride_xn + 
                     c_idx[:, None] * stride_xc +
                     d_idx[None, :] * stride_xd +
                     h_idx[None, :] * stride_xh +
                     w_idx[None, :] * stride_xw)
            
            mask = c_valid[:, None] & spatial_valid[None, :]
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Apply Swish activation
            x_swish = swish_activation_fast(x_vals)
            
            # Update Welford statistics
            for i in range(BLOCK_C):
                if c_valid[i]:
                    # Extract column for this channel
                    col_vals = tl.where(spatial_valid, x_swish[i, :], 0.0)
                    valid_count = tl.sum(spatial_valid)
                    
                    if valid_count > 0:
                        # Online mean/variance update
                        delta = col_vals - mean[i]
                        mean[i] += tl.sum(delta) / (count + valid_count)
                        m2[i] += tl.sum(delta * (col_vals - mean[i]))
            
            count += tl.sum(spatial_valid)
    
    # Final variance computation
    inv_std = 1.0 / tl.sqrt(m2 / count + eps)
    
    # Second pass: apply normalization and activations
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        spatial_valid = spatial_mask & (offs_spatial >= spatial_base) & (offs_spatial < spatial_base + BLOCK_SIZE)
        
        if not tl.sum(spatial_valid):
            continue
            
        for c_base in range(0, group_size, BLOCK_C):
            c_idx = start_c + c_base + channel_offsets
            c_valid = channel_mask & ((c_base + channel_offsets) < group_size)
            
            if not tl.sum(c_valid):
                continue
                
            # Load input
            x_ptrs = (x_ptr + 
                     pid_n * stride_xn + 
                     c_idx[:, None] * stride_xc +
                     d_idx[None, :] * stride_xd +
                     h_idx[None, :] * stride_xh +
                     w_idx[None, :] * stride_xw)
            
            mask = c_valid[:, None] & spatial_valid[None, :]
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Apply Swish activation
            x_swish = swish_activation_fast(x_vals)
            
            # Apply GroupNorm normalization
            # Vectorized normalization using broadcast
            mean_broadcast = mean[:, None]
            inv_std_broadcast = inv_std[:, None]
            x_norm = (x_swish - mean_broadcast) * inv_std_broadcast
            
            # Load weight and bias
            weight = tl.load(weight_ptr + c_idx, mask=c_valid, other=0.0)
            bias = tl.load(bias_ptr + c_idx, mask=c_valid, other=0.0)
            
            # Apply affine transformation
            x_gn = x_norm * weight[:, None] + bias[:, None]
            
            # Apply HardSwish
            x_final = hardswish_activation_fast(x_gn)
            
            # Store result
            out_ptrs = (out_ptr + 
                       pid_n * stride_xn + 
                       c_idx[:, None] * stride_xc +
                       d_idx[None, :] * stride_xd +
                       h_idx[None, :] * stride_xh +
                       w_idx[None, :] * stride_xw)
            
            tl.store(out_ptrs, x_final, mask=mask)


def fused_post_convtranspose_3d(x, weight, bias, groups, eps):
    """
    Optimized fused post-ops for 5D tensors with autotuning
    """
    N, C, D, H, W = x.shape
    
    # Input validation
    if C % groups != 0:
        raise ValueError(f"Number of channels {C} must be divisible by groups {groups}")
    
    # Allocate output
    out = torch.empty_like(x)
    
    # Launch kernel with autotuned configuration
    grid = (N, groups)
    
    fused_swish_groupnorm_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, D, H, W,
        groups, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Optimized fused post-ops (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
        # GroupNorm parameters
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Optimized fused post-ops in Triton
        x = fused_post_convtranspose_3d(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias,
            self.groups,
            self.eps
        )
        
        return x
