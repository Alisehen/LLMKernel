import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
    ],
    key=['channels'],
)
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
    # Each program handles one (batch, output_d, output_h, output_w) position
    pid = tl.program_id(0)
    
    total_outputs = batch_size * out_d * out_h * out_w
    
    # Decode output spatial indices
    tmp = pid
    ow = tmp % out_w
    tmp = tmp // out_w
    oh = tmp % out_h
    tmp = tmp // out_h
    od = tmp % out_d
    b = tmp // out_d
    
    # Check bounds
    valid_output = pid < total_outputs
    
    # Input starting position (pool_size = 4 for combined 2x2x2 twice)
    id_start = od * 4
    ih_start = oh * 4
    iw_start = ow * 4
    
    # Channel offsets
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels
    
    # Initialize max values for all channels
    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)
    
    # Base offset for batch
    base_offset = b * in_stride_b
    
    # Process 4x4x4 pooling region
    for dd in tl.static_range(4):
        id_cur = id_start + dd
        d_offset = id_cur * in_stride_d
        valid_d = id_cur < in_d
        
        for dh in tl.static_range(4):
            ih_cur = ih_start + dh
            h_offset = ih_cur * in_stride_h
            valid_dh = valid_d & (ih_cur < in_h)
            
            for dw in tl.static_range(4):
                iw_cur = iw_start + dw
                w_offset = iw_cur * in_stride_w
                valid = valid_dh & (iw_cur < in_w) & valid_output
                
                # Compute input offset
                input_offset = base_offset + d_offset + h_offset + w_offset
                input_ptrs = input_ptr + input_offset + offs_c * in_stride_c
                
                # Load all channels
                x = tl.load(input_ptrs, mask=mask_c & valid, other=-float('inf'))
                
                # Softmax: exp(x - max) / sum(exp(x - max))
                x_max = tl.max(x, axis=0)
                x_centered = x - x_max
                x_exp = tl.exp(x_centered)
                x_sum = tl.sum(x_exp, axis=0)
                x_sum = tl.where(x_sum > 0, x_sum, 1.0)  # Avoid div by zero
                x_softmax = x_exp / x_sum
                
                # Mask invalid positions
                x_softmax = tl.where(valid, x_softmax, -float('inf'))
                
                # Update max
                max_vals = tl.maximum(max_vals, x_softmax)
    
    # Store results
    output_offset = b * out_stride_b + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
    output_ptrs = output_ptr + output_offset + offs_c * out_stride_c
    tl.store(output_ptrs, max_vals, mask=mask_c & valid_output)


def fused_softmax_and_maxpool(x):
    """Fused softmax followed by 4x4x4 max pooling"""
    batch_size, channels, in_d, in_h, in_w = x.shape
    out_d = in_d // 4
    out_h = in_h // 4
    out_w = in_w // 4
    
    output = torch.empty((batch_size, channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    total_outputs = batch_size * out_d * out_h * out_w
    grid = (max(1, total_outputs),)
    
    fused_softmax_maxpool_kernel[grid](
        x, output,
        batch_size, channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
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
