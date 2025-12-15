import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_relu_group_norm_stats_kernel(
    x_ptr,
    group_sum_ptr,
    group_sum_sq_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    groups,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    """
    Optimized Phase 1: Compute group statistics with improved memory access
    and reduced register pressure
    """
    pid_n = tl.program_id(0)  # Batch index
    pid_g = tl.program_id(1)  # Channel group index
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    
    DHW = D * H * W
    
    # Initialize accumulators in float32 for better precision
    sum_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Process spatial blocks with vectorized loads
    for dhw_offset in range(0, DHW, BLOCK_DHW):
        dhw_idx = tl.arange(0, BLOCK_DHW)
        global_dhw = dhw_offset + dhw_idx
        spatial_mask = global_dhw < DHW
        
        # Compute spatial indices once (reduces register pressure)
        d_idx = global_dhw // (H * W)
        hw_rem = global_dhw % (H * W)
        h_idx = hw_rem // W
        w_idx = hw_rem % W
        
        # Process channels in blocks
        for c_offset in range(0, group_channels, BLOCK_C):
            c_idx = tl.arange(0, BLOCK_C)
            c = group_start + c_offset + c_idx
            c_mask = c < (group_start + group_channels)
            
            # Broadcast masks with minimal memory footprint
            valid_mask = c_mask[:, None] & spatial_mask[None, :]
            
            # Compute offsets efficiently
            offsets = (
                pid_n * stride_n + 
                c[:, None] * stride_c + 
                d_idx[None, :] * stride_d + 
                h_idx[None, :] * stride_h + 
                w_idx[None, :] * stride_w
            )
            
            # Load and apply ReLU
            x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
            x_relu = tl.maximum(x_vals, 0.0)
            
            # Accumulate with direct operations (reduces intermediates)
            sum_val += tl.sum(tl.where(valid_mask, x_relu, 0.0), axis=1)
            x_relu_sq = x_relu * x_relu  # Recompute if register pressure high
            sum_sq += tl.sum(tl.where(valid_mask, x_relu_sq, 0.0), axis=1)
    
    # Efficient reduction across channel block
    total_sum = tl.sum(sum_val)
    total_sum_sq = tl.sum(sum_sq)
    
    # Write group statistics
    out_idx = pid_n * groups + pid_g
    tl.store(group_sum_ptr + out_idx, total_sum)
    tl.store(group_sum_sq_ptr + out_idx, total_sum_sq)


@triton.jit
def fused_relu_group_norm_apply_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    group_sum_ptr,
    group_sum_sq_ptr,
    out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    groups,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
):
    """
    Optimized Phase 2: Apply normalization with flattened spatial dimension
    to improve memory coalescing and reduce register pressure
    """
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    
    # Read group statistics once
    stats_idx = pid_n * groups + pid_g
    group_sum = tl.load(group_sum_ptr + stats_idx)
    group_sum_sq = tl.load(group_sum_sq_ptr + stats_idx)
    
    DHW = D * H * W
    count = tl.cast(group_channels * DHW, tl.float32)
    
    # Compute normalization parameters
    safe_count = tl.maximum(count, 1.0)
    mean = group_sum / safe_count
    variance = tl.maximum(group_sum_sq / safe_count - mean * mean, 0.0)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Process spatial block
    dhw_offset = pid_spatial * BLOCK_DHW
    dhw_idx = tl.arange(0, BLOCK_DHW)
    global_dhw = dhw_offset + dhw_idx
    spatial_mask = global_dhw < DHW
    
    # Compute spatial indices
    d_idx = global_dhw // (H * W)
    hw_rem = global_dhw % (H * W)
    h_idx = hw_rem // W
    w_idx = hw_rem % W
    
    # Process channels in blocks
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = group_start + c_offset + c_idx
        c_mask = c < (group_start + group_channels)
        
        # Broadcast masks
        valid_mask = c_mask[:, None] & spatial_mask[None, :]
        
        # Compute offsets for input
        offsets_in = (
            pid_n * stride_n +
            c[:, None] * stride_c +
            d_idx[None, :] * stride_d +
            h_idx[None, :] * stride_h +
            w_idx[None, :] * stride_w
        )
        
        # Load input and apply ReLU
        x_vals = tl.load(x_ptr + offsets_in, mask=valid_mask, other=0.0)
        x_relu = tl.maximum(x_vals, 0.0)
        
        # Normalize (recomputation reduces register pressure)
        normalized = (x_relu - mean) * inv_std
        
        # Load weight and bias
        weight_vals = tl.load(weight_ptr + c, mask=c_mask, other=0.0)
        bias_vals = tl.load(bias_ptr + c, mask=c_mask, other=0.0)
        
        # Apply affine transformation
        output = normalized * weight_vals[:, None] + bias_vals[:, None]
        
        # Compute output offsets
        offsets_out = (
            pid_n * out_stride_n +
            c[:, None] * out_stride_c +
            d_idx[None, :] * out_stride_d +
            h_idx[None, :] * out_stride_h +
            w_idx[None, :] * out_stride_w
        )
        
        tl.store(out_ptr + offsets_out, output, mask=valid_mask)


def fused_relu_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Optimized wrapper for fused ReLU + GroupNorm
    """
    N, C, D, H, W = x.shape
    assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
    
    out = torch.empty_like(x)
    eps = 1e-5
    group_channels = C // groups
    
    # Phase 1: Statistics with optimized block sizes
    group_sum = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    group_sum_sq = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    
    # Conservative block sizes to avoid register spilling
    BLOCK_C_STATS = min(triton.next_power_of_2(group_channels), 16)  # Reduced for register pressure
    BLOCK_DHW_STATS = 64  # Balanced for memory access
    
    grid_stats = (N, groups)
    fused_relu_group_norm_stats_kernel[grid_stats](
        x, group_sum, group_sum_sq,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        groups, eps,
        BLOCK_C=BLOCK_C_STATS,
        BLOCK_DHW=BLOCK_DHW_STATS,
    )
    
    # Phase 2: Apply with optimized grid
    BLOCK_C_APPLY = min(triton.next_power_of_2(group_channels), 16)  # Conservative
    BLOCK_DHW_APPLY = 64  # Flattened spatial dimension
    
    DHW = D * H * W
    spatial_blocks = (DHW + BLOCK_DHW_APPLY - 1) // BLOCK_DHW_APPLY
    grid_apply = (N, groups, spatial_blocks)
    
    fused_relu_group_norm_apply_kernel[grid_apply](
        x, weight, bias, group_sum, group_sum_sq, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        groups, eps,
        BLOCK_C=BLOCK_C_APPLY,
        BLOCK_DHW=BLOCK_DHW_APPLY,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Optimized Fused ReLU + GroupNorm (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Optimized fused ReLU + GroupNorm
        x = fused_relu_group_norm(x, self.weight, self.bias, self.groups)
        
        return x
