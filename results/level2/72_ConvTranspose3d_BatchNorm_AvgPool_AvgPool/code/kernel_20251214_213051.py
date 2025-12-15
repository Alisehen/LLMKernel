import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_bn_avgpool_kernel_optimized(
    # Pointers to tensors
    x_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr, out_ptr,
    # Tensor dimensions
    N, C, D, H, W,
    D_out, H_out, W_out,
    # Strides for input tensor
    stride_n, stride_c, stride_d, stride_h, stride_w,
    # Strides for output tensor
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    # BatchNorm parameters
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SPATIAL: tl.constexpr,
):
    """
    Optimized fused BatchNorm3d + 4x4x4 AvgPool3d kernel.
    Key optimizations:
    1. Vectorized memory access for BN parameters
    2. Warp-level parallelism for spatial reduction
    3. Better memory coalescing
    4. Minimized redundant calculations
    """
    # Parallelize over channels and batch - grid covers output spatial dimensions
    pid = tl.program_id(0)
    pid_nc = tl.program_id(1)
    
    # Decode batch and channel indices
    num_channels = C
    batch_idx = pid_nc // num_channels
    channel_idx = pid_nc % num_channels
    
    if batch_idx >= N or channel_idx >= C:
        return
    
    # Decode spatial output index using fast integer division
    spatial_size = D_out * H_out * W_out
    num_w_out = W_out
    num_h_out = H_out
    
    # Vectorized processing of multiple spatial outputs per thread block
    spatial_offsets = pid * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    spatial_mask = spatial_offsets < spatial_size
    
    # Precompute strides for fast index calculations
    D_stride = H_out * W_out
    H_stride = W_out
    
    # Decode spatial indices
    d_out = tl.where(spatial_mask, spatial_offsets // D_stride, 0)
    hw_rem = tl.where(spatial_mask, spatial_offsets % D_stride, 0)
    h_out = tl.where(spatial_mask, hw_rem // H_stride, 0)
    w_out = tl.where(spatial_mask, hw_rem % H_stride, 0)
    
    # Load BatchNorm parameters with vectorization
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Compute normalization factor once
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm_factor = gamma * inv_std
    bias_term = beta - mean * norm_factor
    
    # Accumulate over 4x4x4 blocks
    accum = tl.zeros([BLOCK_SPATIAL], dtype=tl.float32)
    count = tl.zeros([BLOCK_SPATIAL], dtype=tl.float32)
    
    # Input spatial start indices
    d_start = d_out * 4
    h_start = h_out * 4
    w_start = w_out * 4
    
    # Unroll and vectorize the 4x4x4 reduction loop
    # Process in groups of 4 for better parallelism
    for d_off in range(0, 4, 2):  # Process 2 depth elements at a time
        d_idx_0 = d_start + d_off
        d_idx_1 = d_start + d_off + 1
        
        for h_off in range(0, 4, 2):  # Process 2 height elements at a time
            h_idx_0 = h_start + h_off
            h_idx_1 = h_start + h_off + 1
            
            for w_off in range(0, 4, 2):  # Process 2 width elements at a time
                w_idx_0 = w_start + w_off
                w_idx_1 = w_start + w_off + 1
                
                # Compute masks for all 8 elements in this 2x2x2 block
                mask_000 = (d_idx_0 < D) & (h_idx_0 < H) & (w_idx_0 < W)
                mask_001 = (d_idx_0 < D) & (h_idx_0 < H) & (w_idx_1 < W)
                mask_010 = (d_idx_0 < D) & (h_idx_1 < H) & (w_idx_0 < W)
                mask_011 = (d_idx_0 < D) & (h_idx_1 < H) & (w_idx_1 < W)
                mask_100 = (d_idx_1 < D) & (h_idx_0 < H) & (w_idx_0 < W)
                mask_101 = (d_idx_1 < D) & (h_idx_0 < H) & (w_idx_1 < W)
                mask_110 = (d_idx_1 < D) & (h_idx_1 < H) & (w_idx_0 < W)
                mask_111 = (d_idx_1 < D) & (h_idx_1 < H) & (w_idx_1 < W)
                
                # Base offsets
                base_offset = (batch_idx * stride_n + channel_idx * stride_c)
                
                # Compute pointer offsets efficiently
                # Using precomputed indices to avoid repeated multiplications
                d_offset_0 = d_idx_0 * stride_d
                d_offset_1 = d_idx_1 * stride_d
                h_offset_0 = h_idx_0 * stride_h
                h_offset_1 = h_idx_1 * stride_h
                w_offset_0 = w_idx_0 * stride_w
                w_offset_1 = w_idx_1 * stride_w
                
                # Load 8 values in pattern for better coalescing
                offsets_000 = base_offset + d_offset_0 + h_offset_0 + w_offset_0
                offsets_001 = base_offset + d_offset_0 + h_offset_0 + w_offset_1
                offsets_010 = base_offset + d_offset_0 + h_offset_1 + w_offset_0
                offsets_011 = base_offset + d_offset_0 + h_offset_1 + w_offset_1
                offsets_100 = base_offset + d_offset_1 + h_offset_0 + w_offset_0
                offsets_101 = base_offset + d_offset_1 + h_offset_0 + w_offset_1
                offsets_110 = base_offset + d_offset_1 + h_offset_1 + w_offset_0
                offsets_111 = base_offset + d_offset_1 + h_offset_1 + w_offset_1
                
                # Vectorized load and batch norm
                x_val_000 = tl.load(x_ptr + offsets_000, mask=mask_000 & spatial_mask, other=0.0)
                x_val_001 = tl.load(x_ptr + offsets_001, mask=mask_001 & spatial_mask, other=0.0)
                x_val_010 = tl.load(x_ptr + offsets_010, mask=mask_010 & spatial_mask, other=0.0)
                x_val_011 = tl.load(x_ptr + offsets_011, mask=mask_011 & spatial_mask, other=0.0)
                x_val_100 = tl.load(x_ptr + offsets_100, mask=mask_100 & spatial_mask, other=0.0)
                x_val_101 = tl.load(x_ptr + offsets_101, mask=mask_101 & spatial_mask, other=0.0)
                x_val_110 = tl.load(x_ptr + offsets_110, mask=mask_110 & spatial_mask, other=0.0)
                x_val_111 = tl.load(x_ptr + offsets_111, mask=mask_111 & spatial_mask, other=0.0)
                
                # Apply BatchNorm: optimized as x * norm_factor + bias_term
                scaled_000 = x_val_000 * norm_factor + bias_term
                scaled_001 = x_val_001 * norm_factor + bias_term
                scaled_010 = x_val_010 * norm_factor + bias_term
                scaled_011 = x_val_011 * norm_factor + bias_term
                scaled_100 = x_val_100 * norm_factor + bias_term
                scaled_101 = x_val_101 * norm_factor + bias_term
                scaled_110 = x_val_110 * norm_factor + bias_term
                scaled_111 = x_val_111 * norm_factor + bias_term
                
                # Accumulate with conditional counting
                accum += tl.where(mask_000, scaled_000, 0.0)
                accum += tl.where(mask_001, scaled_001, 0.0)
                accum += tl.where(mask_010, scaled_010, 0.0)
                accum += tl.where(mask_011, scaled_011, 0.0)
                accum += tl.where(mask_100, scaled_100, 0.0)
                accum += tl.where(mask_101, scaled_101, 0.0)
                accum += tl.where(mask_110, scaled_110, 0.0)
                accum += tl.where(mask_111, scaled_111, 0.0)
                
                count += tl.where(mask_000, 1.0, 0.0)
                count += tl.where(mask_001, 1.0, 0.0)
                count += tl.where(mask_010, 1.0, 0.0)
                count += tl.where(mask_011, 1.0, 0.0)
                count += tl.where(mask_100, 1.0, 0.0)
                count += tl.where(mask_101, 1.0, 0.0)
                count += tl.where(mask_110, 1.0, 0.0)
                count += tl.where(mask_111, 1.0, 0.0)
    
    # Compute average with numerical stability
    result = accum / tl.maximum(count, 1.0)
    
    # Compute output offsets
    out_offsets = (
        batch_idx * out_stride_n + 
        channel_idx * out_stride_c + 
        d_out * out_stride_d + 
        h_out * out_stride_h + 
        w_out * out_stride_w
    )
    
    # Store results with proper masking
    tl.store(out_ptr + out_offsets, result, mask=spatial_mask)


def fused_batch_norm_avgpool_optimized(x, gamma, beta, running_mean, running_var, 
                                      training=False, eps=1e-5, momentum=0.1):
    """
    Optimized fused BatchNorm3d + 4x4x4 AvgPool3d
    
    Args:
        x: Input tensor of shape [N, C, D, H, W]
        gamma: Weight parameter of shape [C]
        beta: Bias parameter of shape [C]
        running_mean: Running mean of shape [C]
        running_var: Running variance of shape [C]
        training: Whether in training mode
        eps: Added to denominator for numerical stability
        momentum: Momentum for updating running statistics
    
    Returns:
        Output tensor of shape [N, C, D//4, H//4, W//4]
    """
    N, C, D, H, W = x.shape
    
    # Output dimensions
    D_out, H_out, W_out = D // 4, H // 4, W // 4
    out_shape = (N, C, D_out, H_out, W_out)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Compute statistics based on training mode
    if training:
        # Compute batch statistics
        mean = x.mean(dim=(0, 2, 3, 4))
        var = x.var(dim=(0, 2, 3, 4), unbiased=False)
        
        # Update running statistics
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1 - momentum).add_(var, alpha=momentum)
    else:
        # Use stored statistics
        mean = running_mean
        var = running_var
    
    # Optimized grid layout
    total_spatial = D_out * H_out * W_out
    BLOCK_SPATIAL = 128  # Optimized for Ada Lovelace
    
    # Grid covers (spatial_blocks, batch*channel)
    num_spatial_blocks = triton.cdiv(total_spatial, BLOCK_SPATIAL)
    num_batch_channel = N * C
    
    grid = (num_spatial_blocks, num_batch_channel)
    
    # Launch optimized kernel
    fused_bn_avgpool_kernel_optimized[grid](
        x, gamma, beta, mean, var, out,
        N, C, D, H, W,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        eps,
        BLOCK_SPATIAL,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized version with fused BatchNorm3d + 2x AvgPool3d (4x4x4 reduction)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        
        # Initialize BatchNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        
        # Set default BatchNorm parameters
        self.eps = 1e-5
        self.momentum = 0.1
    
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Optimized fused BatchNorm3d + AvgPool3d + AvgPool3d in Triton
        x = fused_batch_norm_avgpool_optimized(
            x, self.gamma, self.beta, 
            self.running_mean, self.running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum
        )
        
        return x
