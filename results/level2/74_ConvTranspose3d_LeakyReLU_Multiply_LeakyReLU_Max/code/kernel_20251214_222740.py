import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    VEC_W: tl.constexpr,
):
    """
    Optimized fused kernel: LeakyReLU + Multiply + LeakyReLU + MaxPool3d
    Input: [N, C, D, H, W] -> Output: [N, C, D//2, H//2, W//2]
    Uses vectorized loads for W dimension and optimized memory access pattern
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
    
    # Vectorized indices for C and W dimensions
    offs_c = tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W) * VEC_W
    
    d_start = pid_od * 2
    h_start = pid_oh * 2
    
    # Initialize accumulator with -inf using efficient broadcast
    max_acc = tl.full((BLOCK_C, BLOCK_W * VEC_W), float('-inf'), dtype=tl.float16)
    
    # Pre-calculate negative slope in FP16 for efficiency
    neg_slope_fp16 = tl.semantic_cast(negative_slope, tl.float16)
    
    # Process 2x2x2 pooling window
    for dd in range(2):
        d_idx = d_start + dd
        d_mask = d_idx < D
        
        for dh in range(2):
            h_idx = h_start + dh
            h_mask = h_idx < H
            
            # Combine spatial masks
            spatial_mask = d_mask & h_mask
            
            for c_idx in range(0, C, BLOCK_C):
                c_mask = offs_c < (C - c_idx)
                full_mask = c_mask[:, None] & spatial_mask
                
                # Base pointer calculation
                base_offset = (
                    pid_n * stride_xn +
                    (c_idx + offs_c) * stride_xc +
                    d_idx * stride_xd +
                    h_idx * stride_xh
                )
                
                # Vectorized load for W dimension
                for vw in range(VEC_W):
                    w_indices = offs_w + vw
                    w_mask = w_indices < W
                    final_mask = full_mask & w_mask[None, :]
                    
                    if tl.reduce_or(final_mask) == 0:
                        continue
                    
                    # Load with vectorization
                    x_ptrs = x_ptr + base_offset[:, None] + w_indices[None, :] * stride_xw
                    x_vals = tl.load(x_ptrs, mask=final_mask, other=0.0).to(tl.float16)
                    
                    # Fused operations in registers
                    # Leaky ReLU
                    leaky_mask = x_vals >= 0
                    x_vals = tl.where(leaky_mask, x_vals, x_vals * neg_slope_fp16)
                    
                    # Multiply with channel multiplier
                    multiplier_vals = tl.load(
                        multiplier_ptr + c_idx + offs_c, 
                        mask=c_mask, 
                        other=0.0
                    ).to(tl.float16)[:, None]
                    x_vals = x_vals * multiplier_vals
                    
                    # Second Leaky ReLU
                    leaky_mask2 = x_vals >= 0
                    x_vals = tl.where(leaky_mask2, x_vals, x_vals * neg_slope_fp16)
                    
                    # Update max accumulator
                    current_val = tl.where(final_mask, x_vals, float('-inf'))
                    max_acc = tl.maximum(max_acc, current_val)
    
    # Store final results
    for c_idx in range(0, C, BLOCK_C):
        c_mask = offs_c < (C - c_idx)
        
        # Calculate output indices with vectorization
        w_out_indices = tl.arange(0, BLOCK_W * VEC_W, VEC_W)
        w_mask = w_out_indices < W_out
        
        full_mask = c_mask[:, None] & w_mask[None, :]
        
        if tl.reduce_or(full_mask) == 0:
            continue
        
        # Output pointer calculation
        out_offset = (
            pid_n * stride_out_n +
            (c_idx + offs_c) * stride_out_c +
            pid_od * stride_out_d +
            pid_oh * stride_out_h
        )
        
        out_ptrs = out_ptr + out_offset[:, None] + w_out_indices[None, :] * stride_out_w
        
        # Store with proper masking
        tl.store(out_ptrs, max_acc, mask=full_mask)


def fused_post_convtranspose(x, multiplier, negative_slope=0.2):
    N, C, D, H, W = x.shape
    
    # Convert to FP16 for Tensor Core acceleration
    if x.dtype != torch.float16:
        x = x.half()
    if multiplier.dtype != torch.float16:
        multiplier = multiplier.half()
    
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    
    # Ensure multiplier has correct shape
    multiplier = multiplier.squeeze()
    if multiplier.dim() > 1:
        multiplier = multiplier.view(-1)
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=torch.float16)
    
    # Optimized block sizes for Ada Lovelace
    BLOCK_C = min(triton.next_power_of_2(C), 64)  # Reduced for better register usage
    BLOCK_W = min(triton.next_power_of_2(W_out), 64)
    VEC_W = 2  # Vectorized loads for W dimension
    
    grid = (N, D_out, H_out)
    
    # Launch kernel with optimized num_stages
    fused_leaky_relu_multiply_pool_kernel[grid](
        x, multiplier, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        negative_slope=negative_slope,
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        VEC_W=VEC_W,
        num_warps=8,  # Optimized for 128 SMs
        num_stages=2,  # Start with 2, increase only if needed
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Initialize multiplier in FP16 for efficiency
        self.multiplier = nn.Parameter(torch.randn(out_channels, dtype=torch.float16))
        self.negative_slope = 0.2
        
        # Convert conv_transpose weights to FP16 to match input dtype
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.half()
        if self.conv_transpose.bias is not None:
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.half()
        
    def forward(self, x):
        # Use FP16 throughout for Tensor Core acceleration
        if x.dtype != torch.float16:
            x = x.half()
        
        # Ensure conv weights are in FP16 (double-check in forward)
        weight_fp16 = self.conv_transpose.weight.half()
        bias_fp16 = self.conv_transpose.bias.half() if self.conv_transpose.bias is not None else None
        
        # Perform conv_transpose with FP16 weights
        x = nn.functional.conv_transpose3d(
            x, weight_fp16, bias_fp16,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups
        )
        
        x = fused_post_convtranspose(x, self.multiplier, self.negative_slope)
        return x
