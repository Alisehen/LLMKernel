import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_multiply_gap_kernel_small(
    x_ptr,
    out_ptr,
    multiplier,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    if pid_n >= N or pid_c >= C:
        return
    
    spatial_size = H * W
    inv_spatial_size = 1.0 / spatial_size
    
    hw_idx = tl.arange(0, spatial_size)
    h = hw_idx // W
    w = hw_idx % W
    
    ptr_offset = pid_n * stride_xn + pid_c * stride_xc + h * stride_xh + w * stride_xw
    mask = hw_idx < spatial_size
    spatial_data = tl.load(x_ptr + ptr_offset, mask=mask, other=0.0)
    
    channel_sum = tl.sum(spatial_data)
    result = channel_sum * multiplier * inv_spatial_size
    
    out_offset = pid_n * stride_on + pid_c * stride_oc
    tl.store(out_ptr + out_offset, result)


@triton.jit
def fused_multiply_gap_kernel_large(
    x_ptr,
    out_ptr,
    multiplier,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c_block = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    spatial_size = H * W
    inv_spatial_size = 1.0 / spatial_size
    channel_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    hw_size = spatial_size
    vec_tile_size = BLOCK_HW // VEC_SIZE
    num_tiles = tl.cdiv(hw_size, BLOCK_HW)
    
    for tile_idx in range(num_tiles):
        hw_start = tile_idx * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_size
        
        h = hw_offsets // W
        w = hw_offsets % W
        
        x_base = pid_n * stride_xn + c_offsets[:, None] * stride_xc
        h_offsets = h[None, :] * stride_xh
        w_offsets = w[None, :] * stride_xw
        
        ptr_offsets = x_base + h_offsets + w_offsets
        mask_2d = c_mask[:, None] & hw_mask[None, :]
        
        data = tl.load(x_ptr + ptr_offsets, mask=mask_2d, other=0.0)
        channel_sum += tl.sum(data, axis=1)
    
    result = channel_sum * (multiplier * inv_spatial_size)
    out_base = pid_n * stride_on + c_offsets * stride_oc
    tl.store(out_ptr + out_base, result, mask=c_mask)


@triton.jit
def fused_multiply_gap_kernel_ultra_opt(
    x_ptr,
    out_ptr,
    multiplier,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c_block = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    spatial_size = H * W
    inv_spatial_size = 1.0 / spatial_size
    channel_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    hw_size = spatial_size
    num_tiles = tl.cdiv(hw_size, BLOCK_HW)
    
    stride_xc_local = stride_xc
    stride_xn_local = stride_xn
    W_local = W
    
    for tile_idx in range(num_tiles):
        hw_start = tile_idx * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_size
        
        h = hw_offsets // W_local
        w = hw_offsets % W_local
        
        base_ptr = pid_n * stride_xn_local + c_offsets[:, None] * stride_xc_local
        spatial_offset = h * stride_xh + w * stride_xw
        ptr_offsets = base_ptr + spatial_offset[None, :]
        
        mask_2d = c_mask[:, None] & hw_mask[None, :]
        data = tl.load(x_ptr + ptr_offsets, mask=mask_2d, other=0.0)
        
        # FIXED: Removed tl.where to maintain 1D shape for channel_sum
        channel_sum += tl.sum(data, axis=1)
    
    result = channel_sum * (multiplier * inv_spatial_size)
    out_base = pid_n * stride_on + c_offsets * stride_oc
    tl.store(out_ptr + out_base, result, mask=c_mask)


def fused_multiply_gap(x, multiplier):
    N, C, H, W = x.shape
    spatial_size = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    if spatial_size <= 256:
        BLOCK_C = 1
        grid = (N, C)
        fused_multiply_gap_kernel_small[grid](
            x, out, multiplier,
            N, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_C=BLOCK_C,
        )
    elif spatial_size <= 4096:
        BLOCK_C = min(64, triton.next_power_of_2(C))
        BLOCK_HW = 256
        VEC_SIZE = 4 if x.element_size() == 4 else 8
        
        grid = (N, triton.cdiv(C, BLOCK_C))
        
        fused_multiply_gap_kernel_large[grid](
            x, out, multiplier,
            N, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            VEC_SIZE=VEC_SIZE,
            num_stages=2,
            num_warps=4,
        )
    else:
        BLOCK_C = min(32, triton.next_power_of_2(C))
        BLOCK_HW = 256
        VEC_SIZE = 4 if x.element_size() == 4 else 8
        
        grid = (N, triton.cdiv(C, BLOCK_C))
        
        fused_multiply_gap_kernel_ultra_opt[grid](
            x, out, multiplier,
            N, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            VEC_SIZE=VEC_SIZE,
            num_stages=2,
            num_warps=4,
        )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.multiplier = multiplier
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_multiply_gap(x, self.multiplier)
        return x
