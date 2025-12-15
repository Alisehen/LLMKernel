import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_DHW': 8, 'REDUCE_TILES': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_DHW': 8, 'REDUCE_TILES': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_DHW': 4, 'REDUCE_TILES': 4}, num_warps=4, num_stages=2),
    ],
    key=['C', 'DHW'],
)
@triton.jit
def fused_post_pool_ops_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_out_n, stride_out_d, stride_out_h, stride_out_w,
    BLOCK_C: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
    REDUCE_TILES: tl.constexpr,
):
    """
    Optimized fused kernel with single-pass reduction and vectorization
    """
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    # Reconstruct spatial indices with vectorization
    spatial_base = pid_spatial * BLOCK_DHW
    spatial_idx = spatial_base + tl.arange(0, BLOCK_DHW)
    
    DHW = D * H * W
    spatial_mask = spatial_idx < DHW
    
    # Reconstruct d,h,w indices
    d = spatial_idx // (H * W)
    hw_rem = spatial_idx % (H * W)
    h = hw_rem // W
    w = hw_rem % W
    
    # Initialize reduction buffers in registers for max stability and final max
    max_vals = tl.full((BLOCK_DHW,), float('-inf'), dtype=tl.float32)
    exp_sums = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
    final_max = tl.full((BLOCK_DHW,), float('-inf'), dtype=tl.float32)
    
    # Tile channels for better L2 cache utilization
    for c_start in range(0, C, BLOCK_C * REDUCE_TILES):
        # Precompute bias and x offsets for this tile
        bias_acc = tl.zeros((BLOCK_DHW,), dtype=tl.float32)
        swish_max_acc = tl.full((BLOCK_DHW,), float('-inf'), dtype=tl.float32)
        
        # Process sub-tile
        for tile in range(REDUCE_TILES):
            c_offset = c_start + tile * BLOCK_C
            c_offs = c_offset + tl.arange(0, BLOCK_C)
            c_mask = c_offs < C
            
            # Fix: Replace tl.reduce_or with tl.sum comparison
            if tl.sum(c_mask) > 0:  # Check if any channels are valid
                # Load x values with coalesced access pattern
                x_ptrs = (
                    x_ptr +
                    pid_n * stride_xn +
                    c_offs[:, None] * stride_xc +
                    d[None, :] * stride_xd +
                    h[None, :] * stride_xh +
                    w[None, :] * stride_xw
                )
                x_vals = tl.load(x_ptrs, mask=c_mask[:, None] & spatial_mask[None, :], other=float('-inf'))
                
                # Online softmax update using Welford's algorithm adaptation
                channel_max = tl.max(x_vals, axis=0)
                old_max = max_vals
                max_vals = tl.maximum(max_vals, channel_max)
                
                # Scale previous exponentials and update sum
                scale = tl.exp(old_max - max_vals)
                exp_sums = exp_sums * scale
                
                exp_vals = tl.exp(x_vals - max_vals[None, :])
                exp_sums += tl.sum(exp_vals, axis=0)
                
                # Softmax and swish in same pass (fused operations)
                softmax_vals = exp_vals / tl.maximum(exp_sums[None, :], 1e-12)
                
                # Load bias once per tile
                if tile == 0:
                    bias_vals = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
                
                # Fused subtract and swish
                sub_vals = softmax_vals - bias_vals[:, None]
                # Swish: x * sigmoid(x) - using fast sigmoid approximation
                # 1/(1+exp(-x)) ≈ 0.5 + 0.5 * tanh(0.5*x)
                half_x = sub_vals * 0.5
                sigmoid_vals = 0.5 + 0.5 * (half_x * (1.0 + half_x * half_x * (-0.1666667 + half_x * half_x * 0.00833333)))
                swish_vals = sub_vals * sigmoid_vals
                
                # Update swish max for this tile
                tile_max = tl.max(swish_vals, axis=0)
                swish_max_acc = tl.maximum(swish_max_acc, tile_max)
        
        # Reduce across tiles in this block
        final_max = tl.maximum(final_max, swish_max_acc)
        
        # Early exit if we can guarantee no further contributions
        # (This is approximate but works well for smooth functions)
    
    # Store results with coalesced writes
    out_ptrs = (
        out_ptr +
        pid_n * stride_out_n +
        d * stride_out_d +
        h * stride_out_h +
        w * stride_out_w
    )
    tl.store(out_ptrs, final_max, mask=spatial_mask)


def fused_post_pool_ops(x: torch.Tensor, bias: torch.Tensor):
    N, C, D, H, W = x.shape
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    
    DHW = D * H * W
    
    # Calculate optimal grid - balance SM utilization
    max_grid_n = min(N, 128)  # Limit batch parallelism to improve cache reuse
    grid_n = (N + max_grid_n - 1) // max_grid_n
    
    grid_spatial = triton.cdiv(DHW, 8)  # Start with small spatial blocks
    grid = (grid_n, grid_spatial)
    
    fused_post_pool_ops_kernel[grid](
        x, bias, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        pool_kernel_size,
        pool_stride,
        pool_padding
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool(x)
        x = fused_post_pool_ops(x, self.subtract)
        return x
