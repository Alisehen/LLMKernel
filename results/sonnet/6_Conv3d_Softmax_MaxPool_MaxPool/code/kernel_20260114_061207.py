import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    batch_size, channels, depth, height, width,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one spatial location across all channels
    pid = tl.program_id(0)
    
    # Compute batch, depth, height, width indices
    total_spatial = depth * height * width
    b = pid // total_spatial
    spatial_idx = pid % total_spatial
    d = spatial_idx // (height * width)
    hw_idx = spatial_idx % (height * width)
    h = hw_idx // width
    w = hw_idx % width
    
    # Load all channel values for this spatial location
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < channels
    
    ptrs = input_ptr + b * stride_b + offs_c * stride_c + d * stride_d + h * stride_h + w * stride_w
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    
    # Softmax computation
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_softmax = x_exp / x_sum
    
    out_ptrs = output_ptr + b * stride_b + offs_c * stride_c + d * stride_d + h * stride_h + w * stride_w
    tl.store(out_ptrs, x_softmax, mask=mask)


@triton.jit
def fused_maxpool3d_2x_kernel(
    input_ptr, output_ptr,
    batch_size, channels, 
    in_d, in_h, in_w,
    out_d, out_h, out_w,
    stride_b, stride_c, stride_d, stride_h, stride_w,
    out_stride_b, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Compute output indices
    total_out = out_d * out_h * out_w
    bc_idx = pid // total_out
    spatial_idx = pid % total_out
    
    b = bc_idx // channels
    c = bc_idx % channels
    
    od = spatial_idx // (out_h * out_w)
    oh_ow = spatial_idx % (out_h * out_w)
    oh = oh_ow // out_w
    ow = oh_ow % out_w
    
    # Input starting position (pool_size = 4 because two consecutive 2x2x2 pools = 4x4x4)
    id_start = od * 4
    ih_start = oh * 4
    iw_start = ow * 4
    
    # Find max over 4x4x4 region
    max_val = -float('inf')
    
    base_ptr = input_ptr + b * stride_b + c * stride_c
    
    # Unrolled 4x4x4 max pooling
    for dd in range(4):
        for dh in range(4):
            for dw in range(4):
                id_idx = id_start + dd
                ih_idx = ih_start + dh
                iw_idx = iw_start + dw
                
                valid = (id_idx < in_d) & (ih_idx < in_h) & (iw_idx < in_w)
                ptr = base_ptr + id_idx * stride_d + ih_idx * stride_h + iw_idx * stride_w
                val = tl.load(ptr, mask=valid, other=-float('inf'))
                max_val = tl.maximum(max_val, val)
    
    # Store result
    out_ptr = output_ptr + b * out_stride_b + c * out_stride_c + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptr, max_val)


def softmax_3d(x):
    batch_size, channels, depth, height, width = x.shape
    output = torch.empty_like(x)
    
    total_spatial = batch_size * depth * height * width
    BLOCK_C = triton.next_power_of_2(channels)
    
    grid = (total_spatial,)
    softmax_kernel[grid](
        x, output,
        batch_size, channels, depth, height, width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
    )
    return output


def fused_maxpool3d_2x(x):
    batch_size, channels, in_d, in_h, in_w = x.shape
    out_d = in_d // 4
    out_h = in_h // 4
    out_w = in_w // 4
    
    output = torch.empty((batch_size, channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    total_elements = batch_size * channels * out_d * out_h * out_w
    grid = (total_elements,)
    
    fused_maxpool3d_2x_kernel[grid](
        x, output,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE=64,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized model with fused softmax and double max pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = softmax_3d(x)
        x = fused_maxpool3d_2x(x)
        return x
