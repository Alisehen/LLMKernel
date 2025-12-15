import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_relu_group_norm_kernel(
    x_ptr,  # Input tensor [N, C, D, H, W]
    weight_ptr,  # GroupNorm weight [C]
    bias_ptr,  # GroupNorm bias [C]
    out_ptr,  # Output tensor [N, C, D, H, W]
    N, C, D, H, W,  # Tensor dimensions
    stride_n, stride_c, stride_d, stride_h, stride_w,  # Input strides
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,  # Output strides
    groups,  # Number of groups for GroupNorm
    eps,  # Small epsilon for numerical stability
    GROUP_SIZE_DHW: tl.constexpr,  # Elements per group for reduction
    BLOCK_C: tl.constexpr,  # Block size for channels
    BLOCK_D: tl.constexpr,  # Block size for depth
    BLOCK_H: tl.constexpr,  # Block size for height
    BLOCK_W: tl.constexpr,  # Block size for width
):
    """
    Fused kernel: ReLU + GroupNorm for ConvTranspose3d output
    Processes blocks of [BLOCK_C, BLOCK_D, BLOCK_H, BLOCK_W] elements
    """
    # Program indices for 5D tensor processing
    pid_n = tl.program_id(0)  # Batch dimension
    pid_g = tl.program_id(1)  # Group index (within spatial dimensions)
    
    if pid_n >= N:
        return
    
    # Group normalization parameters
    group_channels = C // groups  # Channels per group (must be divisible)
    
    # Calculate spatial offsets for this thread block
    DHW = D * H * W
    group_id = pid_g
    num_groups_dhw = tl.cdiv(DHW, GROUP_SIZE_DHW)
    
    # Initialize accumulators for group statistics
    sum_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    count = 0
    
    # Loop over channels in this group (C per group)
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = c_offset + c_idx
        
        # Loop over spatial elements in this group
        for dhw_offset in range(0, GROUP_SIZE_DHW, BLOCK_D * BLOCK_H * BLOCK_W):
            # Calculate spatial index within group
            dhw_idx = tl.arange(0, BLOCK_D * BLOCK_H * BLOCK_W)
            global_dhw = group_id * GROUP_SIZE_DHW + dhw_offset + dhw_idx
            
            # Mask for valid dhw_offset
            dhw_offset_mask = dhw_offset < GROUP_SIZE_DHW
            
            if dhw_offset_mask:
                # Unpack spatial indices
                d_idx = global_dhw // (H * W)
                hw_rem = global_dhw % (H * W)
                h_idx = hw_rem // W
                w_idx = hw_rem % W
                
                # Create masks for valid elements with proper broadcasting
                # Channel mask: shape (BLOCK_C, 1)
                c_mask_2d = c[:, None] < group_channels
                # Spatial masks: shape (1, BLOCK_D*BLOCK_H*BLOCK_W)
                d_mask_2d = d_idx[None, :] < D
                h_mask_2d = h_idx[None, :] < H
                w_mask_2d = w_idx[None, :] < W
                dhw_offset_mask_2d = dhw_offset_mask
                
                # Combine masks - all now have shape (BLOCK_C, BLOCK_D*BLOCK_H*BLOCK_W)
                valid_mask = c_mask_2d & d_mask_2d & h_mask_2d & w_mask_2d & dhw_offset_mask_2d
                
                # Compute pointer offsets for input (broadcast to 2D)
                offsets = (
                    pid_n * stride_n + 
                    c[:, None] * stride_c + 
                    d_idx[None, :] * stride_d + 
                    h_idx[None, :] * stride_h + 
                    w_idx[None, :] * stride_w
                )
                
                # Load input values and apply ReLU
                x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
                x_relu = tl.maximum(x_vals, 0.0)
                
                # Accumulate statistics (reduce over spatial dimension)
                sum_val += tl.sum(tl.where(valid_mask, x_relu, 0.0), axis=1)
                sum_sq += tl.sum(tl.where(valid_mask, x_relu * x_relu, 0.0), axis=1)
                count += tl.sum(tl.where(valid_mask, 1.0, 0.0))
    
    # Re-process the same block to apply normalization
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = c_offset + c_idx
        
        # Loop over spatial elements in this group
        for dhw_offset in range(0, GROUP_SIZE_DHW, BLOCK_D * BLOCK_H * BLOCK_W):
            # Calculate spatial index within group
            dhw_idx = tl.arange(0, BLOCK_D * BLOCK_H * BLOCK_W)
            global_dhw = group_id * GROUP_SIZE_DHW + dhw_offset + dhw_idx
            
            # Mask for valid dhw_offset
            dhw_offset_mask = dhw_offset < GROUP_SIZE_DHW
            
            if dhw_offset_mask:
                # Unpack spatial indices
                d_idx = global_dhw // (H * W)
                hw_rem = global_dhw % (H * W)
                h_idx = hw_rem // W
                w_idx = hw_rem % W
                
                # Create masks for valid elements with proper broadcasting
                # Channel mask: shape (BLOCK_C, 1)
                c_mask_2d = c[:, None] < group_channels
                # Spatial masks: shape (1, BLOCK_D*BLOCK_H*BLOCK_W)
                d_mask_2d = d_idx[None, :] < D
                h_mask_2d = h_idx[None, :] < H
                w_mask_2d = w_idx[None, :] < W
                dhw_offset_mask_2d = dhw_offset_mask
                
                # Combine masks
                valid_mask = c_mask_2d & d_mask_2d & h_mask_2d & w_mask_2d & dhw_offset_mask_2d
                
                # Compute pointer offsets for input (broadcast to 2D)
                offsets = (
                    pid_n * stride_n + 
                    c[:, None] * stride_c + 
                    d_idx[None, :] * stride_d + 
                    h_idx[None, :] * stride_h + 
                    w_idx[None, :] * stride_w
                )
                
                # Load input values and apply ReLU
                x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
                x_relu = tl.maximum(x_vals, 0.0)
                
                # Load weight and bias for this channel (broadcast to 2D)
                weight_vals = tl.load(weight_ptr + c, mask=c[:, None] < group_channels, other=0.0)
                bias_vals = tl.load(bias_ptr + c, mask=c[:, None] < group_channels, other=0.0)
                
                # Compute statistics for this block
                local_mean = tl.sum(sum_val) / tl.maximum(count, 1)
                local_var = tl.sum(sum_sq) / tl.maximum(count, 1) - local_mean * local_mean
                
                # Apply group normalization
                norm_factor = 1.0 / tl.sqrt(local_var + eps)
                normalized = (x_relu - local_mean) * norm_factor
                output = normalized * weight_vals + bias_vals
                
                # Store to output
                out_offsets = (
                    pid_n * out_stride_n + 
                    c[:, None] * out_stride_c + 
                    d_idx[None, :] * out_stride_d + 
                    h_idx[None, :] * out_stride_h + 
                    w_idx[None, :] * out_stride_w
                )
                tl.store(out_ptr + out_offsets, output, mask=valid_mask)


def fused_relu_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Wrapper for fused ReLU + GroupNorm kernel
    """
    N, C, D, H, W = x.shape
    
    # Validate dimensions
    assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Group parameters
    eps = 1e-5
    
    # Choose block sizes (powers of 2)
    BLOCK_C = min(triton.next_power_of_2(C // groups), 32)
    BLOCK_D = min(triton.next_power_of_2(D), 8)
    BLOCK_H = min(triton.next_power_of_2(H), 8)
    BLOCK_W = min(triton.next_power_of_2(W), 8)
    
    # Spatial elements per group for processing
    DHW = D * H * W
    GROUP_SIZE_DHW = min(1024, DHW)  # Target ~1024 elements per group
    
    # Compute grid
    num_groups_dhw = (DHW + GROUP_SIZE_DHW - 1) // GROUP_SIZE_DHW
    grid = (N, num_groups_dhw)
    
    # Launch kernel
    fused_relu_group_norm_kernel[grid](
        x, weight, bias, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        groups, eps,
        GROUP_SIZE_DHW=GROUP_SIZE_DHW,
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
