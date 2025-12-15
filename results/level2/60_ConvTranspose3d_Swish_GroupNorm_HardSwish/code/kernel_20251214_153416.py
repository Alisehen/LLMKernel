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
    
    # Welford's algorithm for online mean/variance computation
    channel_offsets = tl.arange(0, BLOCK_C)
    channel_mask = channel_offsets < group_size
    
    # Initialize accumulators
    mean = tl.zeros((BLOCK_C,), dtype=tl.float32)
    m2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Process spatial blocks for statistics
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        # Compute spatial indices for current block
        spatial_offsets = spatial_base + offs_spatial
        spatial_valid = spatial_offsets < DHW
        
        # Precompute spatial indices
        spatial_idx = tl.where(spatial_valid, spatial_offsets, 0)
        d_idx = tl.where(spatial_valid, spatial_idx // (H * W), 0)
        hw_idx = tl.where(spatial_valid, spatial_idx % (H * W), 0)
        h_idx = tl.where(spatial_valid, hw_idx // W, 0)
        w_idx = tl.where(spatial_valid, hw_idx % W, 0)
        
        # Process channels in vectorized manner
        for c_base in range(0, group_size, BLOCK_C):
            c_idx = start_c + c_base + channel_offsets
            c_valid = channel_mask & ((c_base + channel_offsets) < group_size)
            
            # Only process if there are valid elements in this block
            any_valid = tl.sum(c_valid) > 0
            
            # Use conditional load/store with masks
            if tl.sum(spatial_valid) > 0 and any_valid:
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
                
                # Update Welford statistics - use vectorized operations
                # Count valid spatial elements for each channel
                spatial_count = tl.sum(spatial_valid).to(tl.float32)
                count_update = tl.where(c_valid, spatial_count, 0.0)
                
                # Compute mean update
                col_sum = tl.sum(x_swish, axis=1)
                delta = tl.where(c_valid, col_sum - mean * spatial_count, 0.0)
                mean_update = tl.where(c_valid, delta / (count + count_update + 1e-10), 0.0)
                
                # Update mean and m2
                new_mean = mean + mean_update
                delta2 = x_swish - new_mean[:, None]
                m2_update = tl.where(mask, delta2 * delta2, 0.0)
                m2_new = m2 + tl.sum(m2_update, axis=1)
                
                # Apply updates where valid
                mean = tl.where(c_valid, new_mean, mean)
                m2 = tl.where(c_valid, m2_new, m2)
                count = tl.where(c_valid, count + count_update, count)
    
    # Final variance computation
    inv_std = 1.0 / tl.sqrt(m2 / tl.maximum(count, 1.0) + eps)
    
    # Second pass: apply normalization and activations
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        # Compute spatial indices for current block
        spatial_offsets = spatial_base + offs_spatial
        spatial_valid = spatial_offsets < DHW
        
        # Precompute spatial indices
        spatial_idx = tl.where(spatial_valid, spatial_offsets, 0)
        d_idx = tl.where(spatial_valid, spatial_idx // (H * W), 0)
        hw_idx = tl.where(spatial_valid, spatial_idx % (H * W), 0)
        h_idx = tl.where(spatial_valid, hw_idx // W, 0)
        w_idx = tl.where(spatial_valid, hw_idx % W, 0)
        
        for c_base in range(0, group_size, BLOCK_C):
            c_idx = start_c + c_base + channel_offsets
            c_valid = channel_mask & ((c_base + channel_offsets) < group_size)
            
            # Only process if there are valid elements
            if tl.sum(spatial_valid) > 0 and tl.sum(c_valid) > 0:
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
