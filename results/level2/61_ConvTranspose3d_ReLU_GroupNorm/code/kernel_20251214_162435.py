import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_DHW': 128, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_DHW': 128, 'VECTOR_SIZE': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 16, 'BLOCK_DHW': 256, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W', 'groups']
)
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
    VECTOR_SIZE: tl.constexpr,
    num_spatial_blocks: tl.constexpr,
    num_c_blocks: tl.constexpr,
):
    """Phase 1: Compute group statistics with vectorized loads and optimal tiling"""
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_combined = tl.program_id(2)
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    DHW = D * H * W
    
    pid_c_block = pid_combined % num_c_blocks
    pid_spatial_block = pid_combined // num_c_blocks
    
    if pid_c_block >= num_c_blocks or pid_spatial_block >= num_spatial_blocks:
        return
    
    c_offset = pid_c_block * BLOCK_C
    dhw_offset = pid_spatial_block * BLOCK_DHW
    
    # Use vectorized indices for better memory throughput
    c_idx = tl.arange(0, BLOCK_C)
    dhw_idx_base = tl.arange(0, BLOCK_DHW) * VECTOR_SIZE
    dhw_idx = dhw_idx_base[:, None] + tl.arange(0, VECTOR_SIZE)[None, :]
    
    c = group_start + c_offset + c_idx
    global_dhw = dhw_offset + dhw_idx
    
    # Masks with vectorization
    c_mask = c < (group_start + group_channels)
    spatial_mask = global_dhw < DHW
    valid_mask = c_mask[:, None, None] & spatial_mask[None, :, :]
    
    # Precompute spatial indices (optimized for vectorization)
    HW = H * W
    d_idx = global_dhw // HW
    hw_rem = global_dhw % HW
    h_idx = hw_rem // W
    w_idx = hw_rem % W
    
    # Vectorized offsets calculation
    offsets = (
        pid_n * stride_n +
        c[:, None, None] * stride_c +
        d_idx[None, :, :] * stride_d +
        h_idx[None, :, :] * stride_h +
        w_idx[None, :, :] * stride_w
    )
    
    # Vectorized load with masking
    x_vals = tl.load(x_ptr + offsets, mask=valid_mask, other=0.0)
    x_relu = tl.maximum(x_vals, 0.0)
    
    # Fast reduction using vectorized operations
    sum_val = tl.sum(x_relu, axis=[1, 2])
    x_relu_sq = x_relu * x_relu
    sum_sq = tl.sum(x_relu_sq, axis=[1, 2])
    
    # Reduce across channels in this block
    block_sum = tl.sum(sum_val)
    block_sum_sq = tl.sum(sum_sq)
    
    # Atomic reduction with check to avoid unnecessary atomics
    stats_idx = pid_n * groups + pid_g
    if tl.abs(block_sum) > 1e-8:
        tl.atomic_add(group_sum_ptr + stats_idx, block_sum)
    if tl.abs(block_sum_sq) > 1e-8:
        tl.atomic_add(group_sum_sq_ptr + stats_idx, block_sum_sq)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_DHW': 128, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 32, 'BLOCK_DHW': 128, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 16, 'BLOCK_DHW': 256, 'VECTOR_SIZE': 4}, num_warps=4, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W', 'groups']
)
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
    BLOCK_DHW: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """Phase 2: Apply normalization with vectorized loads/stores and optimized caching"""
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    
    if pid_n >= N or pid_g >= groups:
        return
    
    group_channels = C // groups
    group_start = pid_g * group_channels
    
    # Read group statistics once per block with caching hint
    stats_idx = pid_n * groups + pid_g
    group_sum = tl.load(group_sum_ptr + stats_idx, cache_modifier=".cg")
    group_sum_sq = tl.load(group_sum_sq_ptr + stats_idx, cache_modifier=".cg")
    
    DHW = D * H * W
    count = tl.cast(group_channels * DHW, tl.float32)
    
    # Fast normalization parameters
    mean = tl.math.fast_div(group_sum, tl.maximum(count, 1.0))
    var = tl.maximum(tl.math.fast_div(group_sum_sq, tl.maximum(count, 1.0)) - mean * mean, 0.0)
    inv_std = tl.math.rsqrt(var + eps)
    
    # Process spatial block with vectorization
    dhw_offset = pid_spatial * BLOCK_DHW
    dhw_idx_base = tl.arange(0, BLOCK_DHW) * VECTOR_SIZE
    dhw_idx = dhw_idx_base[:, None] + tl.arange(0, VECTOR_SIZE)[None, :]
    global_dhw = dhw_offset + dhw_idx
    spatial_mask = global_dhw < DHW
    
    HW = H * W
    d_idx = global_dhw // HW
    hw_rem = global_dhw % HW
    h_idx = hw_rem // W
    w_idx = hw_rem % W
    
    # Process channels in blocks (optimized for register usage)
    for c_offset in range(0, group_channels, BLOCK_C):
        c_idx = tl.arange(0, BLOCK_C)
        c = group_start + c_offset + c_idx
        c_mask = c < (group_start + group_channels)
        
        # Precompute output offsets for vectorized store
        offsets_out = (
            pid_n * out_stride_n +
            c[:, None, None] * out_stride_c +
            d_idx[None, :, :] * out_stride_d +
            h_idx[None, :, :] * out_stride_h +
            w_idx[None, :, :] * out_stride_w
        )
        
        valid_mask = c_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Compute input offsets
        offsets_in = (
            pid_n * stride_n +
            c[:, None, None] * stride_c +
            d_idx[None, :, :] * stride_d +
            h_idx[None, :, :] * stride_h +
            w_idx[None, :, :] * stride_w
        )
        
        # Vectorized load with caching hint
        x_vals = tl.load(x_ptr + offsets_in, mask=valid_mask, other=0.0, cache_modifier=".ca")
        x_relu = tl.maximum(x_vals, 0.0)
        
        # Fused normalization and affine transform
        normalized = (x_relu - mean) * inv_std
        
        # Load weight and bias once per channel block with caching
        weight_vals = tl.load(weight_ptr + c, mask=c_mask, other=0.0, cache_modifier=".cg")
        bias_vals = tl.load(bias_ptr + c, mask=c_mask, other=0.0, cache_modifier=".cg")
        
        # Apply affine transform with broadcasting
        output = normalized * weight_vals[:, None, None] + bias_vals[:, None, None]
        
        # Vectorized store
        tl.store(out_ptr + offsets_out, output, mask=valid_mask, cache_modifier=".cg")


def fused_relu_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    """Optimized wrapper for fused ReLU + GroupNorm with autotune"""
    N, C, D, H, W = x.shape
    assert C % groups == 0, f"Channels {C} must be divisible by groups {groups}"
    
    out = torch.empty_like(x)
    eps = 1e-5
    group_channels = C // groups
    DHW = D * H * W
    
    # Phase 1: Statistics computation
    group_sum = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    group_sum_sq = torch.zeros(N * groups, dtype=torch.float32, device=x.device)
    
    # Dynamic block sizing with autotune
    num_c_blocks = (group_channels + 31) // 32  # Conservative start
    num_spatial_blocks = (DHW + 127) // 128
    num_blocks = num_c_blocks * num_spatial_blocks
    
    grid_stats = (N, groups, num_blocks)
    
    fused_relu_group_norm_stats_kernel[grid_stats](
        x, group_sum, group_sum_sq,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        groups, eps,
        num_spatial_blocks=num_spatial_blocks,
        num_c_blocks=num_c_blocks,
    )
    
    # Phase 2: Apply normalization
    spatial_blocks = (DHW + 127) // 128  # Conservative start
    grid_apply = (N, groups, spatial_blocks)
    
    fused_relu_group_norm_apply_kernel[grid_apply](
        x, weight, bias, group_sum, group_sum_sq, out,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        groups, eps,
    )
    
    return out


class ModelNew(nn.Module):
    """ConvTranspose3d + Optimized Fused ReLU + GroupNorm"""
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Optimized fused ReLU + GroupNorm
        x = fused_relu_group_norm(x, self.weight, self.bias, self.groups)
        
        return x
