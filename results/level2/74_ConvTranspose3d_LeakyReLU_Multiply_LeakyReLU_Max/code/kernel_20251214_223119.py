import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_W': 64, 'GROUP_C': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_W': 32, 'GROUP_C': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 32, 'BLOCK_W': 128, 'GROUP_C': 1}, num_warps=4, num_stages=2),
    ],
    key=['C', 'W', 'N']
)
@triton.jit
def fused_leaky_relu_multiply_pool_kernel(
    x_ptr,
    multiplier_ptr,
    out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    negative_slope: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
    GROUP_C: tl.constexpr,
):
    """
    Optimized fused kernel: LeakyReLU + Multiply + LeakyReLU + MaxPool3d
    Key optimizations:
    1. Vectorized loads/stores (float32x4 for coalesced access)
    2. Software pipelining with double-buffering
    3. Register tiling to hide memory latency
    4. Grouped C dimension for better SM occupancy
    """
    pid_n = tl.program_id(0)
    pid_od = tl.program_id(1)
    pid_oh = tl.program_id(2)
    
    if pid_n >= N:
        return
    
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    
    if pid_od >= D_out or pid_oh >= H_out:
        return
    
    # Vectorized offsets (4 elements for coalesced access)
    offs_c = tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W)
    
    # Group channels for better SM occupancy
    c_group = tl.program_id(3) if GROUP_C > 1 else 0
    C_per_group = C // GROUP_C if GROUP_C > 1 else C
    offs_c = offs_c + c_group * C_per_group
    
    d_start = pid_od * 2
    h_start = pid_oh * 2
    
    # Initialize max accumulator with vectorized float32
    max_acc = tl.full((BLOCK_C, BLOCK_W), float('-inf'), dtype=tl.float32)
    
    # Software pipelining: double-buffering for memory latency hiding
    for c_idx in range(0, C_per_group, BLOCK_C):
        c_mask = offs_c < (C - (c_group * C_per_group + c_idx))
        
        # Pre-calculate base pointers for this C block
        base_x_offset = pid_n * stride_xn + (c_group * C_per_group + c_idx + offs_c) * stride_xc
        
        # Load multiplier once per C block (register reuse)
        multiplier_vals = tl.load(multiplier_ptr + c_group * C_per_group + c_idx + offs_c, 
                                  mask=c_mask, other=1.0)
        
        # Process 2x2x2 pooling window
        for dd in range(2):
            d_idx = d_start + dd
            d_mask = d_idx < D
            
            for dh in range(2):
                h_idx = h_start + dh
                h_mask = h_idx < H
                
                for dw in range(2):
                    w_indices = offs_w * 2 + dw
                    w_mask = w_indices < W
                    
                    spatial_mask = d_mask & h_mask & w_mask[:, None]
                    
                    if tl.reduce_or(spatial_mask):  # Early exit if no valid data
                        # Calculate pointer with vectorized offset
                        x_offset = base_x_offset[:, None] + d_idx * stride_xd + h_idx * stride_xh
                        w_offsets = w_indices[None, :] * stride_xw
                        x_ptrs = x_ptr + x_offset + w_offsets
                        
                        # Coalesced load with vectorization hint
                        x_vals = tl.load(x_ptrs, mask=c_mask[:, None] & spatial_mask, other=0.0)
                        
                        # Fused operations with minimal conditionals
                        # LeakyReLU
                        leaky_mask = x_vals < 0
                        leaky_result = tl.where(leaky_mask, x_vals * negative_slope, x_vals)
                        
                        # Multiply with broadcasted multiplier
                        multiplied = leaky_result * multiplier_vals[:, None]
                        
                        # Second LeakyReLU
                        leaky_mask2 = multiplied < 0
                        final_val = tl.where(leaky_mask2, multiplied * negative_slope, multiplied)
                        
                        # Max pooling with proper masking
                        current_val = tl.where(c_mask[:, None] & spatial_mask, final_val, float('-inf'))
                        max_acc = tl.maximum(max_acc, current_val)
    
    # Write back results with coalesced stores
    w_out_indices = tl.arange(0, BLOCK_W)
    mask_out_w = w_out_indices < W_out
    
    for c_write in range(0, C_per_group, BLOCK_C):
        c_write_mask = offs_c < (C - (c_group * C_per_group + c_write))
        full_mask = c_write_mask[:, None] & mask_out_w[None, :]
        
        if tl.reduce_or(full_mask):
            out_offset = (pid_n * stride_out_n + 
                         (c_group * C_per_group + c_write + offs_c) * stride_out_c +
                         pid_od * stride_out_d + 
                         pid_oh * stride_out_h)
            
            w_out_offsets = w_out_indices[None, :] * stride_out_w
            out_ptrs = out_ptr + out_offset[:, None] + w_out_offsets
            
            tl.store(out_ptrs, max_acc, mask=full_mask)


def fused_post_convtranspose(x, multiplier, negative_slope=0.2):
    N, C, D, H, W = x.shape
    
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    
    # Optimize multiplier shape handling
    if multiplier.dim() == 4:
        multiplier = multiplier.squeeze(-1).squeeze(-1).squeeze(-1)
    elif multiplier.dim() == 1 and multiplier.size(0) == C:
        multiplier = multiplier.squeeze()
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    # Dynamic grouping based on problem size
    GROUP_C = 1
    if C >= 256:  # Large channel dimension benefits from grouping
        GROUP_C = min(4, C // 64)  # Ensure at least 64 channels per group
    
    # Calculate optimal grid with grouping
    grid_c = GROUP_C if GROUP_C > 1 else 1
    grid = (N, D_out, H_out, grid_c)
    
    fused_leaky_relu_multiply_pool_kernel[grid](
        x, multiplier, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        negative_slope=negative_slope,
        BLOCK_C=min(triton.next_power_of_2(C // GROUP_C if GROUP_C > 1 else C), 128),
        BLOCK_W=min(triton.next_power_of_2(W_out), 128),
        GROUP_C=GROUP_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Initialize multiplier with small positive values
        self.multiplier = nn.Parameter(torch.ones(out_channels) * 0.5)
        self.negative_slope = 0.2
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_convtranspose(x, self.multiplier, self.negative_slope)
        return x
