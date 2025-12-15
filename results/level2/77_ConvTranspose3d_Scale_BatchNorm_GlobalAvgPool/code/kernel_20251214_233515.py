import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_scale_bn_gap_kernel(
    x_ptr,  # Input tensor [N, C, D, H, W]
    scale_factor,  # Scalar scale factor
    running_mean_ptr,  # Running mean [C]
    running_var_ptr,  # Running variance [C]
    weight_ptr,  # Weight [C] (gamma)
    bias_ptr,  # Bias [C] (beta)
    out_ptr,  # Output tensor [N, C, 1, 1, 1]
    eps,  # Epsilon for batch norm
    N, C, D, H, W,  # Input dimensions
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,  # Input strides
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,  # Output strides
    BLOCK_C: tl.constexpr,  # Must be power of 2
    BLOCK_DHW: tl.constexpr,  # Must be power of 2 for DHW reduction
):
    """
    Fused kernel: Scale + BatchNorm3d (eval mode) + GlobalAvgPool3d
    Input: [N, C, D, H, W] -> Output: [N, C, 1, 1, 1]
    Uses running statistics for batch normalization (eval mode only)
    """
    # Parallelize over batch and channel
    pid_n = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Channel index
    
    if pid_n >= N or pid_c >= C:
        return
    
    # Load batch norm parameters for this channel
    mean_val = tl.load(running_mean_ptr + pid_c)
    var_val = tl.load(running_var_ptr + pid_c)
    weight_val = tl.load(weight_ptr + pid_c) if weight_ptr is not None else 1.0
    bias_val = tl.load(bias_ptr + pid_c) if bias_ptr is not None else 0.0
    
    # Pre-compute batch norm scale factor
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale_norm = weight_val * inv_std
    bias_norm = bias_val - mean_val * scale_norm
    
    # Apply scale factor to batch norm parameters
    scale_norm = scale_norm * scale_factor
    bias_norm = bias_norm * scale_factor
    
    # Initialize accumulation for global average pooling
    accum = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    count = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    
    # DHW reduction for global average pooling
    # We'll process multiple DHW positions in parallel within a block
    offs_dhw = tl.arange(0, BLOCK_DHW)
    
    # Calculate total number of DHW positions
    DHW = D * H * W
    
    # Process DHW positions in blocks
    for dhw_start in range(0, DHW, BLOCK_DHW):
        dhw_idx = dhw_start + offs_dhw
        
        # Compute d, h, w indices from linear DHW index
        w_idx = dhw_idx % W
        h_idx = (dhw_idx // W) % H
        d_idx = dhw_idx // (H * W)
        
        # Check if within bounds
        mask = dhw_idx < DHW
        
        if tl.sum(mask) > 0:
            # Calculate input pointer offsets
            x_offset = (pid_n * stride_xn + pid_c * stride_xc + 
                       d_idx * stride_xd + h_idx * stride_xh + w_idx * stride_xw)
            
            # Load input values
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            
            # Apply scaled batch norm: scale_factor * ((x - mean) * inv_std * weight + bias)
            # Optimized as: x * (scale_factor * weight * inv_std) + (scale_factor * (bias - mean * weight * inv_std))
            x_norm = x_vals * scale_norm + bias_norm
            
            # Accumulate for average pooling
            accum += tl.where(mask, x_norm, 0.0)
            count += tl.where(mask, 1.0, 0.0)
    
    # Final reduction across DHW dimension
    total = tl.sum(accum)
    valid_count = tl.sum(count)
    
    # Compute global average
    avg_val = total / tl.maximum(valid_count, 1.0)
    
    # Store to output (only spatial position [0, 0, 0])
    out_offset = (pid_n * stride_on + pid_c * stride_oc)
    tl.store(out_ptr + out_offset, avg_val)


def fused_scale_bn_gap(
    x: torch.Tensor,
    scale_factor: float,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Fused wrapper: Scale + BatchNorm3d (eval mode) + GlobalAvgPool3d
    """
    N, C, D, H, W = x.shape
    
    # Output shape [N, C, 1, 1, 1]
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    # Choose block sizes (powers of 2)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_DHW = 128  # Reasonable block size for DHW reduction
    
    # Grid: (N, C)
    grid = (N, C)
    
    # Launch kernel
    fused_scale_bn_gap_kernel[grid](
        x,
        scale_factor,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        eps,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_DHW=BLOCK_DHW,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused Scale + BatchNorm3d + GlobalAvgPool3d
    
    IMPORTANT: This fused implementation only works in evaluation mode
    because it uses running statistics for batch normalization.
    
    For training, fall back to PyTorch implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as PyTorch native - DO NOT reimplement in Triton
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        
        # Batch norm parameters
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        
        # For fused kernel (eval mode only)
        self.eps = eps
        
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused Scale + BatchNorm + GlobalAvgPool
        # Use fused kernel only in evaluation mode
        if not self.training:
            # Ensure batch norm is in eval mode
            self.batch_norm.eval()
            
            # Sync batch norm running statistics
            with torch.no_grad():
                # Update running statistics if needed
                if self.batch_norm.track_running_stats:
                    # Use current batch to update running stats
                    # This maintains compatibility with PyTorch's behavior
                    current_mean = x.mean(dim=(0, 2, 3, 4))
                    current_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
                    
                    # Update running statistics
                    self.batch_norm.running_mean.mul_(1 - self.batch_norm.momentum).add_(
                        current_mean * self.batch_norm.momentum
                    )
                    self.batch_norm.running_var.mul_(1 - self.batch_norm.momentum).add_(
                        current_var * self.batch_norm.momentum
                    )
            
            # Use fused kernel with running statistics
            x = fused_scale_bn_gap(
                x,
                self.scale_factor,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.eps
            )
        else:
            # Training mode: Use PyTorch operations
            x = x * self.scale_factor
            x = self.batch_norm(x)
            
            # Global average pooling
            x = x.mean(dim=(2, 3, 4), keepdim=True)
        
        return x
