import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_scale_pool_bias_scale_kernel(
    x_ptr, out_ptr,
    scale1_ptr, bias_ptr, scale2_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    POOL_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fused: Scale1 + AvgPool3d(2) + BiasAdd + Scale2
    Optimized for Ada Lovelace (4090) with 3D grid and better memory access patterns
    
    Grid layout: (N, C_BLOCKS, D_BLOCKS * HW_BLOCKS)
    Each thread block processes: [BLOCK_C] channels × [BLOCK_D] depth × [BLOCK_HW] height×width
    """
    # 3D grid indices
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)
    
    if pid_n >= N:
        return
    
    # Output dimensions
    D_out = D // POOL_K
    H_out = H // POOL_K
    W_out = W // POOL_K
    HW_out = H_out * W_out
    
    # Block sizes for spatial dimensions
    D_BLOCKS = tl.cdiv(D_out, BLOCK_D)
    HW_BLOCKS = tl.cdiv(HW_out, BLOCK_HW)
    
    # Decode spatial block indices
    pid_d_block = pid_dhw // HW_BLOCKS
    pid_hw_block = pid_dhw % HW_BLOCKS
    
    # Channel offsets
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    
    # Depth offsets (output space)
    d_out_offs = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_out_offs < D_out
    
    # Height×Width offsets (linearized)
    hw_offs = pid_hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW_out
    
    # Load parameters once per thread block
    scale1 = tl.load(scale1_ptr)
    scale2 = tl.load(scale2_ptr)
    bias_vals = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
    
    # Process each depth position
    for d_idx in tl.static_range(BLOCK_D):
        d_out = d_out_offs[d_idx]
        if d_mask[d_idx]:
            # Convert linear HW to 2D indices
            h_out = hw_offs // W_out
            w_out = hw_offs % W_out
            
            # Input starting positions
            d_in_start = d_out * POOL_K
            h_in_start = h_out * POOL_K
            w_in_start = w_out * POOL_K
            
            # Accumulator for pooling (2x2x2 = 8 elements)
            pool_sum = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
            
            # Unrolled pooling over 8 neighbors
            for dd in tl.static_range(POOL_K):
                d_in = d_in_start + dd
                for dh in tl.static_range(POOL_K):
                    h_in = h_in_start + dh
                    for dw in tl.static_range(POOL_K):
                        w_in = w_in_start + dw
                        
                        # Vectorized load for all channels and HW positions
                        x_offset = (
                            pid_n * stride_xn +
                            tl.reshape(c_offs, (BLOCK_C, 1)) * stride_xc +
                            d_in * stride_xd +
                            tl.reshape(h_in, (1, BLOCK_HW)) * stride_xh +
                            tl.reshape(w_in, (1, BLOCK_HW)) * stride_xw
                        )
                        
                        # Load with broadcasting for spatial dimensions
                        x_vals = tl.load(
                            x_ptr + tl.broadcast_to(x_offset, (BLOCK_C, BLOCK_HW)),
                            mask=tl.broadcast_to(c_mask[:, None] & hw_mask[None, :] & (d_in < D) & (h_in < H) & (w_in < W), (BLOCK_C, BLOCK_HW)),
                            other=0.0
                        )
                        pool_sum += x_vals * scale1
            
            # Average pooling and final computation
            pool_avg = pool_sum / (POOL_K * POOL_K * POOL_K)
            result = (pool_avg + tl.reshape(bias_vals, (BLOCK_C, 1))) * scale2
            
            # Store result
            out_offset = (
                pid_n * stride_on +
                tl.reshape(c_offs, (BLOCK_C, 1)) * stride_oc +
                d_out * stride_od +
                tl.reshape(h_out, (1, BLOCK_HW)) * stride_oh +
                tl.reshape(w_out, (1, BLOCK_HW)) * stride_ow
            )
            
            tl.store(
                out_ptr + tl.broadcast_to(out_offset, (BLOCK_C, BLOCK_HW)),
                result,
                mask=tl.broadcast_to(c_mask[:, None] & hw_mask[None, :], (BLOCK_C, BLOCK_HW))
            )


def fused_post_convtranspose(x, scale1, bias, scale2):
    """Wrapper for fused kernel with optimized grid configuration"""
    N, C, D, H, W = x.shape
    D_out, H_out, W_out = D // 2, H // 2, W // 2
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Optimized block sizes for Ada Lovelace
    BLOCK_C = min(triton.next_power_of_2(C), 128)  # Reduced for better occupancy
    BLOCK_D = 4  # Process multiple depth positions
    BLOCK_HW = 16  # Process 16 HW positions together
    
    # Grid calculation
    C_BLOCKS = (C + BLOCK_C - 1) // BLOCK_C
    D_BLOCKS = (D_out + BLOCK_D - 1) // BLOCK_D
    HW_out = H_out * W_out
    HW_BLOCKS = (HW_out + BLOCK_HW - 1) // BLOCK_HW
    
    grid = (N, C_BLOCKS, D_BLOCKS * HW_BLOCKS)
    
    # Launch kernel
    fused_scale_pool_bias_scale_kernel[grid](
        x, out,
        scale1, bias, scale2,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        POOL_K=2,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        BLOCK_HW=BLOCK_HW,
    )
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused: Scale1 + AvgPool3d(2) + BiasAdd + Scale2 (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_convtranspose(x, self.scale1, self.bias, self.scale2)
        return x
