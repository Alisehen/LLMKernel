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
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Fused kernel: LeakyReLU + Multiply + LeakyReLU + MaxPool3d
    Input: [N, C, D, H, W] -> Output: [N, C, D//2, H//2, W//2]
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
    
    offs_c = tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W)
    
    d_start = pid_od * 2
    h_start = pid_oh * 2
    w_starts = offs_w * 2
    
    max_acc = tl.full((BLOCK_C, BLOCK_W), float('-inf'), dtype=tl.float32)
    
    for dd in range(2):
        d_idx = d_start + dd
        d_mask = d_idx < D
        
        for dh in range(2):
            h_idx = h_start + dh
            h_mask = h_idx < H
            
            for dw in range(2):
                w_indices = w_starts + dw
                w_mask = w_indices < W
                
                spatial_mask = d_mask & h_mask & w_mask
                
                for c_idx in range(0, C, BLOCK_C):
                    c_mask = offs_c < (C - c_idx)
                    full_mask = c_mask[:, None] & spatial_mask[None, :]
                    
                    # Fixed: Use tl.sum to check if any mask is true
                    if tl.sum(tl.where(full_mask, 1, 0)) == 0:
                        continue
                    
                    base_ptr = (
                        pid_n * stride_xn +
                        (c_idx + offs_c) * stride_xc +
                        d_idx * stride_xd +
                        h_idx * stride_xh
                    )
                    
                    x_ptrs = base_ptr[:, None] + w_indices[None, :] * stride_xw
                    x_vals = tl.load(x_ptrs, mask=full_mask, other=0.0)
                    
                    x_vals = tl.where(x_vals >= 0, x_vals, x_vals * negative_slope)
                    
                    multiplier_vals = tl.load(multiplier_ptr + c_idx + offs_c, mask=c_mask, other=0.0)
                    x_vals = x_vals * multiplier_vals[:, None]
                    
                    x_vals = tl.where(x_vals >= 0, x_vals, x_vals * negative_slope)
                    
                    current_mask = tl.where(full_mask, x_vals, float('-inf'))
                    max_acc = tl.maximum(max_acc, current_mask)
    
    w_out_indices = tl.arange(0, BLOCK_W)
    mask_out_w = w_out_indices < W_out
    
    for c_idx in range(0, C, BLOCK_C):
        c_mask = offs_c < (C - c_idx)
        full_mask = c_mask[:, None] & mask_out_w[None, :]
        
        # Fixed: Use tl.sum to check if any mask is true
        if tl.sum(tl.where(full_mask, 1, 0)) == 0:
            continue
        
        out_base = (
            pid_n * stride_out_n +
            (c_idx + offs_c) * stride_out_c +
            pid_od * stride_out_d +
            pid_oh * stride_out_h
        )
        out_ptrs = out_base[:, None] + w_out_indices[None, :] * stride_out_w
        
        tl.store(out_ptrs, max_acc, mask=full_mask)


def fused_post_convtranspose(x, multiplier, negative_slope=0.2):
    N, C, D, H, W = x.shape
    
    D_out = D // 2
    H_out = H // 2
    W_out = W // 2
    
    multiplier = multiplier.squeeze()
    if multiplier.dim() == 4:
        multiplier = multiplier.squeeze(-1).squeeze(-1).squeeze(-1)
    
    out = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    BLOCK_C = min(triton.next_power_of_2(C), 128)
    BLOCK_W = min(triton.next_power_of_2(W_out), 128)
    
    grid = (N, D_out, H_out)
    
    fused_leaky_relu_multiply_pool_kernel[grid](
        x, multiplier, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        negative_slope=negative_slope,
        BLOCK_C=BLOCK_C,
        BLOCK_D=1,
        BLOCK_H=1,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(out_channels))
        self.negative_slope = 0.2
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_convtranspose(x, self.multiplier, self.negative_slope)
        return x
