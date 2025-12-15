import torch
import torch.nn as nn
import triton
import triton.language as tl

# Optimized configs based on Ada Lovelace architecture and register pressure
# Using smaller block sizes to prevent register spilling while maintaining occupancy
configs = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2),  # Conservative fallback
]

@triton.autotune(configs=configs, key=['N', 'C_out', 'H', 'W'])
@triton.jit
def fused_conv_hardswish_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, C_in, H, W,
    C_out, K_H, K_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Optimize grid partitioning for better occupancy
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N * (H - K_H + 1) * (W - K_W + 1), BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    H_out = H - K_H + 1
    W_out = W - K_W + 1
    DHW = H_out * W_out
    
    # Offsets in output space
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = offs_m // DHW
    spatial_idx = offs_m % DHW
    oh_idx = spatial_idx // W_out
    ow_idx = spatial_idx % W_out
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks for boundaries
    mask_m = offs_m < N * DHW
    ih_base = oh_idx
    iw_base = ow_idx
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Precompute some indices to reduce register pressure
    n_idx_stride = tl.where(mask_m, n_idx * stride_xn, 0)
    oh_idx_stride = tl.where(mask_m, oh_idx * stride_oh, 0)
    ow_idx_stride = tl.where(mask_m, ow_idx * stride_ow, 0)
    
    # Main convolution loop
    for ic in range(C_in):
        # Precompute input channel offset once
        x_ic_offset = ic * stride_xc
        w_ic_offset = ic * stride_wic
        
        for kh in range(K_H):
            ih = ih_base + kh
            ih_mask = (ih >= 0) & (ih < H)
            ih_offset = tl.where(ih_mask & mask_m, ih * stride_xh, 0)
            
            for kw in range(K_W):
                iw = iw_base + kw
                iw_mask = (iw >= 0) & (iw < W)
                iw_offset = tl.where(iw_mask & mask_m, iw * stride_xw, 0)
                
                # Combined mask for this position
                pos_mask = mask_m & ih_mask & iw_mask
                
                # Load input with single value per thread (broadcast across BLOCK_N)
                x_ptrs = x_ptr + n_idx_stride + x_ic_offset + ih_offset + iw_offset
                x_val = tl.load(x_ptrs, mask=pos_mask, other=0.0)
                
                # Load weights - vector load across output channels
                w_ptrs = w_ptr + w_ic_offset + kh * stride_wkh + kw * stride_wkw + offs_n * stride_woc
                w_vals = tl.load(w_ptrs, mask=offs_n < C_out, other=0.0)
                
                # FMA: use explicit expansion for better register allocation
                # x_val is broadcast across BLOCK_N dimension
                acc += x_val[:, None] * w_vals[None, :]
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < C_out, other=0.0)
        acc += bias[None, :]
    
    # Optimized HardSwish + ReLU fusion with reduced intermediate storage
    # Original: shifted = acc + 3.0, relu6 = min(max(shifted, 0), 6), hardswish = acc * relu6 / 6.0, out = max(hardswish, 0)
    # Fused to minimize register usage:
    # 1. Compute shifted once and reuse for relu6
    # 2. Fuse division into multiplication (multiply by 1/6)
    # 3. Compute final ReLU in place
    
    # Use fused operations to reduce register pressure
    shifted = acc + 3.0
    # Compute relu6 with min/max chaining
    relu6 = tl.where(shifted > 6.0, 6.0, tl.where(shifted < 0.0, 0.0, shifted))
    
    # Compute HardSwish with division by multiplication
    hardswish = acc * relu6 * (1.0 / 6.0)
    
    # Final ReLU in place (no new allocation)
    out = tl.where(hardswish > 0.0, hardswish, 0.0)
    
    # Store output
    mask_out = mask_m[:, None] & (offs_n[None, :] < C_out)
    
    # Compute output pointers with precomputed strides
    out_ptrs = out_ptr + (
        tl.where(mask_m, n_idx, 0)[:, None] * stride_on + 
        offs_n[None, :] * stride_oc + 
        tl.where(mask_m, oh_idx, 0)[:, None] * stride_oh + 
        tl.where(mask_m, ow_idx, 0)[:, None] * stride_ow
    )
    
    tl.store(out_ptrs, out, mask=mask_out)


def fused_conv_hardswish_relu(x, weight, bias, stride=1, padding=0, dilation=1):
    N, C_in, H, W = x.shape
    C_out, _, K_H, K_W = weight.shape
    
    H_out = H - K_H + 1
    W_out = W - K_W + 1
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Use 1D grid for better load balancing
    P = N * H_out * W_out
    grid = lambda META: (triton.cdiv(P, META['BLOCK_M']) * triton.cdiv(C_out, META['BLOCK_N']),)
    
    weight_contig = weight.contiguous()
    bias_contig = bias.contiguous() if bias is not None else None
    
    fused_conv_hardswish_relu_kernel[grid](
        x, weight_contig, bias_contig, out,
        N, C_in, H, W,
        C_out, K_H, K_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight_contig.stride(0), weight_contig.stride(1),
        weight_contig.stride(2), weight_contig.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.kernel_size = kernel_size
        self.K_H, self.K_W = kernel_size
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.K_H, self.K_W)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))
    
    def forward(self, x):
        return fused_conv_hardswish_relu(x, self.weight, self.bias)
