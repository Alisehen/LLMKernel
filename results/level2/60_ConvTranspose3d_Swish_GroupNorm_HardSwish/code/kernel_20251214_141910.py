import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def swish_activation(x):
    """Swish activation: x * sigmoid(x)"""
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    return x * sigmoid


@triton.jit
def hardswish_activation(x):
    """HardSwish activation: x * relu6(x + 3) / 6"""
    # Compute relu6(x + 3) - clip between 0 and 6
    x_plus_3 = x + 3.0
    relu6 = tl.where(x_plus_3 < 0.0, 0.0, 
                    tl.where(x_plus_3 > 6.0, 6.0, x_plus_3))
    return x * relu6 / 6.0


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
    Fused kernel: Swish + GroupNorm + HardSwish
    Input: [N, C, D, H, W], groups for GroupNorm
    Output: [N, C, D, H, W]
    
    Each program processes one group in one sample across all spatial positions.
    We use program_id(0) for batch, program_id(1) for groups.
    """
    pid_n = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index
    
    if pid_n >= N or pid_g >= groups:
        return
    
    # Calculate group parameters
    group_size = C // groups
    start_c = pid_g * group_size
    
    # Offsets for channels in this group
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < group_size
    
    # Precompute spatial dimensions
    DHW = D * H * W
    offs_spatial = tl.arange(0, BLOCK_SIZE)
    
    # Step 1: Compute mean and variance for the group
    # Initialize as 1D tensors
    sum_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count = 0.0
    
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        spatial_offs = spatial_base + offs_spatial
        spatial_mask = spatial_offs < DHW
        
        # Reconstruct spatial indices
        d_idx = spatial_offs // (H * W)
        h_idx = (spatial_offs % (H * W)) // W
        w_idx = spatial_offs % W
        
        # Accumulate across channels in the group using vectorization
        for c in range(0, group_size, BLOCK_C):
            c_offs = c + offs_c
            c_idx = start_c + c_offs
            c_valid = c_mask & (c_offs < group_size)
            
            # Check if any channel in this block is valid using masking
            has_valid = tl.sum(c_valid) > 0
            
            # Use conditional execution with masks instead of continue
            if has_valid:
                # Calculate pointer offsets with broadcasting
                x_ptrs = (x_ptr + 
                         pid_n * stride_xn + 
                         c_idx[:, None] * stride_xc +
                         d_idx[None, :] * stride_xd +
                         h_idx[None, :] * stride_xh +
                         w_idx[None, :] * stride_xw)
                
                # Load and apply Swish with masking
                mask = c_valid[:, None] & spatial_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
                x_swish = swish_activation(x_vals)
                
                # Sum over spatial dimension to get 1D tensor
                spatial_sum = tl.sum(x_swish, axis=1)
                spatial_sum_sq = tl.sum(x_swish * x_swish, axis=1)
                
                # Accumulate using 1D masks and 1D results
                sum_val += tl.where(c_valid, spatial_sum, 0.0)
                sum_sq += tl.where(c_valid, spatial_sum_sq, 0.0)
        
        count += tl.sum(spatial_mask) * group_size
    
    # Compute mean and variance for the group
    # Sum across channels in the group
    total_sum = tl.sum(sum_val)
    total_sq = tl.sum(sum_sq)
    group_mean = total_sum / (count + eps)
    group_var = total_sq / (count + eps) - group_mean * group_mean
    inv_std = 1.0 / tl.sqrt(group_var + eps)
    
    # Step 2: Apply normalization and activations
    for spatial_base in range(0, DHW, BLOCK_SIZE):
        spatial_offs = spatial_base + offs_spatial
        spatial_mask = spatial_offs < DHW
        
        # Reconstruct spatial indices
        d_idx = spatial_offs // (H * W)
        h_idx = (spatial_offs % (H * W)) // W
        w_idx = spatial_offs % W
        
        # Process channels in vectorized manner
        for c in range(0, group_size, BLOCK_C):
            c_offs = c + offs_c
            c_idx = start_c + c_offs
            c_valid = c_mask & (c_offs < group_size)
            
            # Check if any channel in this block is valid using masking
            has_valid = tl.sum(c_valid) > 0
            
            # Use conditional execution with masks instead of continue
            if has_valid:
                # Load input with broadcasting
                x_ptrs = (x_ptr + 
                         pid_n * stride_xn + 
                         c_idx[:, None] * stride_xc +
                         d_idx[None, :] * stride_xd +
                         h_idx[None, :] * stride_xh +
                         w_idx[None, :] * stride_xw)
                
                mask = c_valid[:, None] & spatial_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
                
                # Apply Swish
                x_swish = swish_activation(x_vals)
                
                # Apply GroupNorm normalization
                x_norm = (x_swish - group_mean) * inv_std
                
                # Load weight and bias for these channels
                weight = tl.load(weight_ptr + c_idx, mask=c_valid, other=0.0)
                bias = tl.load(bias_ptr + c_idx, mask=c_valid, other=0.0)
                
                # Apply affine transformation with broadcasting
                x_gn = x_norm * weight[:, None] + bias[:, None]
                
                # Apply HardSwish
                x_final = hardswish_activation(x_gn)
                
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
    Fused: Swish + GroupNorm + HardSwish for 5D tensors
    """
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    
    # Ensure C is divisible by groups
    if C % groups != 0:
        raise ValueError(f"Number of channels {C} must be divisible by groups {groups}")
    
    # Choose block sizes
    group_size = C // groups
    BLOCK_C = triton.next_power_of_2(min(group_size, 64))
    BLOCK_SIZE = 128  # Spatial block size
    
    # Grid: (batch, groups)
    grid = (N, groups)
    
    fused_swish_groupnorm_hardswish_kernel[grid](
        x, weight, bias, out,
        N, C, D, H, W,
        groups, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused post-ops (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, bias=bias
        )
        
        # GroupNorm parameters - we'll use these in the fused kernel
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused post-ops in Triton
        x = fused_post_convtranspose_3d(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias,
            self.groups,
            self.eps
        )
        
        return x
