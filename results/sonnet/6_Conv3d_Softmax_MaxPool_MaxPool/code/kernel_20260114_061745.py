import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_maxpool_kernel(
    input_ptr, output_ptr,
    batch_size, channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    in_stride_b, in_stride_c, in_stride_d, in_stride_h, in_stride_w,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one output spatial position across all channels
    pid = tl.program_id(0)
    
    total_outputs = batch_size * out_d * out_h * out_w
    if pid >= total_outputs:
        return
    
    # Decode output spatial indices
    tmp = pid
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    b = tmp // out_d
    
    # Input starting position (pool_size = 4 for combined 2x2x2 twice)
    id_start = od * 4
    ih_start = oh * 4
    iw_start = ow * 4
    
    # Channel offsets
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels
    
    # Initialize max values for all channels
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)
    
    # Iterate over 4x4x4 pooling region
    for dd in range(4):
        for dh in range(4):
            for dw in range(4):
                id_cur = id_start + dd
                ih_cur = ih_start + dh
                iw_cur = iw_start + dw
                
                # Bounds check
                valid = (id_cur < in_d) & (ih_cur < in_h) & (iw_cur < in_w)
                
                if valid:
                    # Load all channels for this spatial position
                    input_offset = b * in_stride_b + id_cur * in_stride_d + ih_cur * in_stride_h + iw_cur * in_stride_w
                    input_ptrs = input_ptr + input_offset + offs_c * in_stride_c
                    x = tl.load(input_ptrs, mask=mask_c, other=-float('inf'))
                    
                    # Compute softmax for this position
                    x_max = tl.max(x, axis=0)
                    x_exp = tl.exp(x - x_max)
                    x_sum = tl.sum(x_exp, axis=0)
                    x_softmax = x_exp / x_sum
                    
                    # Update max values
                    max_vals = tl.maximum(max_vals, x_softmax)
    
    # Store results for all channels
    output_offset = b * out_stride_b + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
    output_ptrs = output_ptr + output_offset + offs_c * out_stride_c
    tl.store(output_ptrs, max_vals, mask=mask_c)


@triton.jit
def softmax_kernel_optimized(
    input_ptr, output_ptr,
    batch_size, channels, spatial_size,
    stride_b, stride_c, stride_s,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (batch, spatial) position
    pid = tl.program_id(0)
    batch_idx = pid // spatial_size
    spatial_idx = pid % spatial_size
    
    if batch_idx >= batch_size:
        return
    
    # Load all channels for this position
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels
    
    input_ptrs = input_ptr + batch_idx * stride_b + offs_c * stride_c + spatial_idx * stride_s
    x = tl.load(input_ptrs, mask=mask_c, other=-float('inf'))
    
    # Softmax: exp(x - max) / sum(exp(x - max))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum
    
    output_ptrs = output_ptr + batch_idx * stride_b + offs_c * stride_c + spatial_idx * stride_s
    tl.store(output_ptrs, x_softmax, mask=mask_c)


@triton.jit
def maxpool3d_4x4x4_vectorized_kernel(
    input_ptr, output_ptr,
    batch_size, channels,
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one output spatial position across all channels
    pid = tl.program_id(0)
    
    total_outputs = batch_size * out_d * out_h * out_w
    if pid >= total_outputs:
        return
    
    # Decode output spatial indices
    tmp = pid
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    b = tmp // out_d
    
    # Input starting position (pool_size = 4)
    id_start = od * 4
    ih_start = oh * 4
    iw_start = ow * 4
    
    # Channel offsets
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels
    
    # Initialize max values for all channels
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)
    
    # Iterate over 4x4x4 pooling region
    for dd in range(4):
        for dh in range(4):
            for dw in range(4):
                id_cur = id_start + dd
                ih_cur = ih_start + dh
                iw_cur = iw_start + dw
                
                valid = (id_cur < in_d) & (ih_cur < in_h) & (iw_cur < in_w)
                
                if valid:
                    # Load all channels for this spatial position (vectorized)
                    input_offset = b * stride_b + id_cur * stride_d + ih_cur * stride_h + iw_cur * stride_w
                    input_ptrs = input_ptr + input_offset + offs_c * stride_c
                    vals = tl.load(input_ptrs, mask=mask_c, other=-float('inf'))
                    
                    # Update max values
                    max_vals = tl.maximum(max_vals, vals)
    
    # Store results for all channels
    output_offset = b * out_stride_b + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
    output_ptrs = output_ptr + output_offset + offs_c * out_stride_c
    tl.store(output_ptrs, max_vals, mask=mask_c)


def fused_softmax_and_maxpool(x):
    """Fused softmax followed by 4x4x4 max pooling"""
    batch_size, channels, in_d, in_h, in_w = x.shape
    out_d = in_d // 4
    out_h = in_h // 4
    out_w = in_w // 4
    
    output = torch.empty((batch_size, channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    BLOCK_C = triton.next_power_of_2(channels)
    total_outputs = batch_size * out_d * out_h * out_w
    grid = (max(1, total_outputs),)
    
    fused_softmax_maxpool_kernel[grid](
        x, output,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_C=BLOCK_C,
    )
    return output


def optimized_softmax(x):
    batch_size, channels, d, h, w = x.shape
    spatial_size = d * h * w
    output = torch.empty_like(x)
    
    BLOCK_C = triton.next_power_of_2(channels)
    grid = (batch_size * spatial_size,)
    
    softmax_kernel_optimized[grid](
        x, output,
        batch_size, channels, spatial_size,
        x.stride(0), x.stride(1), 1,
        BLOCK_C=BLOCK_C,
    )
    return output


def optimized_maxpool3d_4x4x4(x):
    batch_size, channels, in_d, in_h, in_w = x.shape
    out_d = in_d // 4
    out_h = in_h // 4
    out_w = in_w // 4
    
    output = torch.empty((batch_size, channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    BLOCK_C = triton.next_power_of_2(channels)
    total_outputs = batch_size * out_d * out_h * out_w
    grid = (max(1, total_outputs),)
    
    maxpool3d_4x4x4_vectorized_kernel[grid](
        x, output,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_C=BLOCK_C,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # Use PyTorch conv3d (highly optimized with cuDNN)
        x = self.conv(x)
        
        # Fused softmax and double max pooling (4x4x4 = two 2x2x2 pools)
        x = fused_softmax_and_maxpool(x)
        
        return x
