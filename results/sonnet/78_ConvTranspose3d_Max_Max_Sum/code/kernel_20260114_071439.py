import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_pool3d_kernel(
    input_ptr, output_ptr,
    batch, channels, in_d, in_h, in_w,
    out_d, out_h, out_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    stride_bn, stride_bc, stride_bd, stride_bh, stride_bw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * channels * out_d * out_h * out_w
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    n = idx // (channels * out_d * out_h * out_w)
    rem = idx % (channels * out_d * out_h * out_w)
    c = rem // (out_d * out_h * out_w)
    rem = rem % (out_d * out_h * out_w)
    od = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w
    
    max_val = -1e20
    
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                id = od * stride_d + kd
                ih = oh * stride_h + kh
                iw = ow * stride_w + kw
                
                valid = (id < in_d) & (ih < in_h) & (iw < in_w)
                
                input_offset = (n * stride_bn + c * stride_bc + 
                               id * stride_bd + ih * stride_bh + iw * stride_bw)
                val = tl.load(input_ptr + input_offset, mask=mask & valid, other=-1e20)
                max_val = tl.maximum(max_val, val)
    
    output_offset = (n * stride_on + c * stride_oc + 
                    od * stride_od + oh * stride_oh + ow * stride_ow)
    tl.store(output_ptr + output_offset, max_val, mask=mask)

@triton.jit
def sum_channels_kernel(
    input_ptr, output_ptr,
    batch, channels, depth, height, width,
    stride_bn, stride_bc, stride_bd, stride_bh, stride_bw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * depth * height * width
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    n = idx // (depth * height * width)
    rem = idx % (depth * height * width)
    d = rem // (height * width)
    rem = rem % (height * width)
    h = rem // width
    w = rem % width
    
    sum_val = 0.0
    for c in range(channels):
        input_offset = (n * stride_bn + c * stride_bc + 
                       d * stride_bd + h * stride_bh + w * stride_bw)
        val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
        sum_val += val
    
    output_offset = (n * stride_on + 0 * stride_oc + 
                    d * stride_od + h * stride_oh + w * stride_ow)
    tl.store(output_ptr + output_offset, sum_val, mask=mask)

def max_pool3d_triton(x, kernel_size, stride=None):
    if stride is None:
        stride = kernel_size
    
    batch, channels, in_d, in_h, in_w = x.shape
    
    if isinstance(kernel_size, int):
        kernel_d = kernel_h = kernel_w = kernel_size
    else:
        kernel_d, kernel_h, kernel_w = kernel_size
    
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
    
    out_d = (in_d - kernel_d) // stride_d + 1
    out_h = (in_h - kernel_h) // stride_h + 1
    out_w = (in_w - kernel_w) // stride_w + 1
    
    output = torch.empty((batch, channels, out_d, out_h, out_w), 
                         device=x.device, dtype=x.dtype)
    
    total_elements = batch * channels * out_d * out_h * out_w
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    max_pool3d_kernel[grid](
        x, output,
        batch, channels, in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def sum_channels_triton(x):
    batch, channels, depth, height, width = x.shape
    output = torch.empty((batch, 1, depth, height, width), 
                         device=x.device, dtype=x.dtype)
    
    total_elements = batch * depth * height * width
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    sum_channels_kernel[grid](
        x, output,
        batch, channels, depth, height, width,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = max_pool3d_triton(x, kernel_size=2)
        x = max_pool3d_triton(x, kernel_size=3)
        x = sum_channels_triton(x)
        return x
