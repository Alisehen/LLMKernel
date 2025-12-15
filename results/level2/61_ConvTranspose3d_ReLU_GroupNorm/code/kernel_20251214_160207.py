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
    Phase 1: Compute group statistics (sum and sum_sq) across entire groups
    """
    pid_n = tl.program_id(0)  # Batch index
    pid_g = tl.program_id(1)  # Channel group index
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    group_end = group_start + group_channels
    
    DHW = D * H * W
    
    # Initialize accumulators for this group
    sum_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Loop over channels in this group
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = group_start + c_offset + c_idx
        c_mask = c < group_end
        
        # Loop over spatial positions
        for dhw_offset in range(0, DHW, BLOCK_DHW):
            dhw_idx = tl.arange(0, BLOCK_DHW)
            global_dhw = dhw_offset + dhw_idx
            spatial_mask = global_dhw < DHW
            
            # Unpack spatial indices
            d_idx = global_dhw // (H * W)
            hw_rem = global_dhw % (H * W)
            h_idx = hw_rem // W
            w_idx = hw_rem % W
            
            # Broadcast masks
            valid_mask = c_mask[:, None] & spatial_mask[None, :]
            
            # Load input and apply ReLU
            offsets = (
                pid_n * stride_n + 
                c[:, None] * stride_c + 
                d_idx[None, :] * stride_d + 
                h_idx[None, :] * stride_h + 
                w_idx[None, :] * stride_w
            )
            
            x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
            x_relu = tl.maximum(x_vals, 0.0)
            
            # Accumulate statistics
            sum_val += tl.sum(tl.where(valid_mask, x_relu, 0.0), axis=1)
            sum_sq += tl.sum(tl.where(valid_mask, x_relu * x_relu, 0.0), axis=1)
    
    # Reduce across channel block
    total_sum = tl.sum(sum_val)
    total_sum_sq = tl.sum(sum_sq)
    count = tl.cast(group_channels * DHW, tl.float32)
    
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
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Phase 2: Apply ReLU + GroupNorm using precomputed group statistics
    """
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    
    # Read group statistics
    stats_idx = pid_n * groups + pid_g
    group_sum = tl.load(group_sum_ptr + stats_idx)
    group_sum_sq = tl.load(group_sum_sq_ptr + stats_idx)
    count = tl.cast(group_channels * D * H * W, tl.float32)
    
    # Compute group mean and variance
    safe_count = tl.maximum(count, 1.0)
    mean = group_sum / safe_count
    variance = tl.maximum(group_sum_sq / safe_count - mean * mean, 0.0)
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Calculate number of spatial blocks
    blocks_d = (D + BLOCK_D - 1) // BLOCK_D
    blocks_h = (H + BLOCK_H - 1) // BLOCK_H
    blocks_w = (W + BLOCK_W - 1) // BLOCK_W
    total_spatial_blocks = blocks_d * blocks_h * blocks_w
    
    if pid_spatial >= total_spatial_blocks:
        return
    
    # Decompose spatial block index
    block_d = pid_spatial // (blocks_h * blocks_w)
    rem = pid_spatial % (blocks_h * blocks_w)
    block_h = rem // blocks_w
    block_w = rem % blocks_w
    
    # Process spatial block
    d_offset = block_d * BLOCK_D
    h_offset = block_h * BLOCK_H
    w_offset = block_w * BLOCK_W
    
    d_idx = d_offset + tl.arange(0, BLOCK_D)
    h_idx = h_offset + tl.arange(0, BLOCK_H)
    w_idx = w_offset + tl.arange(0, BLOCK_W)
    
    d_mask = d_idx < D
    h_mask = h_idx < H
    w_mask = w_idx < W
    
    # Process channels in this group
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = group_start + c_offset + c_idx
        c_mask = c < (group_start + group_channels)
        
        # Broadcast masks
        d_mask_2d = d_mask[None, :, None, None]
        h_mask_2d = h_mask[None, None, :, None]
        w_mask_2d = w_mask[None, None, None, :]
        c_mask_2d = c_mask[:, None, None, None]
        valid_mask = c_mask_2d & d_mask_2d & h_mask_2d & w_mask_2d
        
        # Compute offsets
        offsets = (
            pid_n * stride_n +
            c[:, None, None, None] * stride_c +
            d_idx[None, :, None, None] * stride_d +
            h_idx[None, None, :, None] * stride_h +
            w_idx[None, None, None, :] * stride_w
        )
        
        # Load and apply ReLU
        x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
        x_relu = tl.maximum(x_vals, 0.0)
        
        # Normalize
        normalized = (x_relu - mean) * inv_std
        
        # Load weight and bias
        weight_vals = tl.load(weight_ptr + c, mask=c_mask, other=0.0)
        bias_vals = tl.load(bias_ptr + c, mask=c_mask, other=0.0)
        
        # Apply affine transformation
        output = normalized * weight_vals[:, None, None, None] + bias_vals[:, None, None, None]
        
        # Store output
        out_offsets = (
            pid_n * out_stride_n +
            c[:, None, None, None] * out_stride_c +
            d_idx[None, :, None, None] * out_stride_d +
            h_idx[None, None, :, None] * out_stride_h +
            w_idx[None, None, None, :] * out_stride_w
        )
        tl.store(out_ptr + out_offsets, output, mask=valid_mask)


def fused_relu_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Wrapper for fused ReLU + GroupNorm using two-phase kernel
    """
    N, C, D, H, W = x.shape
    
    # Validate dimensions
    assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Group parameters
    eps = 1e-5
    group_channels = C // groups
    
    # Phase 1: Compute group statistics
    group_sum = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    group_sum_sq = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    
    # Launch statistics kernel
    BLOCK_C_STATS = min(triton.next_power_of_2(group_channels), 32)
    BLOCK_DHW_STATS = 128  # Process 128 spatial positions at once
    
    grid_stats = (N, groups)
    fused_relu_group_norm_stats_kernel[grid_stats](
        x, group_sum, group_sum_sq,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        groups, eps,
        BLOCK_C=BLOCK_C_STATS,
        BLOCK_DHW=BLOCK_DHW_STATS,
    )
    
    # Phase 2: Apply normalization
    BLOCK_C = min(triton.next_power_of_2(group_channels), 32)
    BLOCK_D = min(triton.next_power_of_2(D), 8)
    BLOCK_H = min(triton.next_power_of_2(H), 8)
    BLOCK_W = min(triton.next_power_of_2(W), 8)
    
    # Calculate 3D grid with flattened spatial dimension
    blocks_d = (D + BLOCK_D - 1) // BLOCK_D
    blocks_h = (H + BLOCK_H - 1) // BLOCK_H
    blocks_w = (W + BLOCK_W - 1) // BLOCK_W
    total_spatial_blocks = blocks_d * blocks_h * blocks_w
    
    grid_apply = (N, groups, total_spatial_blocks)
    
    fused_relu_group_norm_apply_kernel[grid_apply](
        x, weight, bias, group_sum, group_sum_sq, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        groups, eps,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused ReLU + GroupNorm (Triton)
    
    Uses PyTorch's native ConvTranspose3d (complex index mapping) and fuses
    the subsequent ReLU and GroupNorm operations in Triton for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # Keep ConvTranspose3d as PyTorch native - DO NOT reimplement in Triton
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        
        # GroupNorm parameters (learnable)
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        
        # Initialize weights
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W)
        """
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused ReLU + GroupNorm in Triton
        x = fused_relu_group_norm(x, self.weight, self.bias, self.groups)
        
        return x
