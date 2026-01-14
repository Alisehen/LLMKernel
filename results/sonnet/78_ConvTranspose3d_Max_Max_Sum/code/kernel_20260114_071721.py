import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pool_pool_sum_kernel(
    input_ptr, output_ptr,
    batch, channels, in_d, in_h, in_w,
    out_d, out_h, out_w,
    stride_bn, stride_bc, stride_bd, stride_bh, stride_bw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * out_d * out_h * out_w
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    n = idx // (out_d * out_h * out_w)
    rem = idx % (out_d * out_h * out_w)
    od = rem // (out_h * out_w)
    rem = rem % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w
    
    # First max pool (kernel=2, stride=2)
    # Compute intermediate dimensions after first pool
    inter_d = (in_d - 2) // 2 + 1
    inter_h = (in_h - 2) // 2 + 1
    inter_w = (in_w - 2) // 2 + 1
    
    # Second max pool (kernel=3, stride=3)
    # Output position maps to intermediate position
    inter_od = od * 3
    inter_oh = oh * 3
    inter_ow = ow * 3
    
    # Accumulate sum across all channels
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for c in range(channels):
        # Second pool: iterate over 3x3x3 kernel in intermediate space
        max_val_second = tl.full([BLOCK_SIZE], -1e20, dtype=tl.float32)
        
        for kd2 in range(3):
            for kh2 in range(3):
                for kw2 in range(3):
                    inter_id = inter_od + kd2
                    inter_ih = inter_oh + kh2
                    inter_iw = inter_ow + kw2
                    
                    # Check if intermediate position is valid
                    valid_inter = (inter_id < inter_d) & (inter_ih < inter_h) & (inter_iw < inter_w)
                    
                    # First pool: compute max over 2x2x2 kernel in input space
                    # Intermediate position maps to input position
                    in_id_base = inter_id * 2
                    in_ih_base = inter_ih * 2
                    in_iw_base = inter_iw * 2
                    
                    max_val_first = tl.full([BLOCK_SIZE], -1e20, dtype=tl.float32)
                    
                    for kd1 in range(2):
                        for kh1 in range(2):
                            for kw1 in range(2):
                                in_id = in_id_base + kd1
                                in_ih = in_ih_base + kh1
                                in_iw = in_iw_base + kw1
                                
                                valid_input = (in_id < in_d) & (in_ih < in_h) & (in_iw < in_w) & mask & valid_inter
                                
                                input_offset = (n * stride_bn + c * stride_bc + 
                                               in_id * stride_bd + in_ih * stride_bh + in_iw * stride_bw)
                                val = tl.load(input_ptr + input_offset, mask=valid_input, other=-1e20)
                                max_val_first = tl.maximum(max_val_first, val)
                    
                    # Use first pool result for second pool
                    max_val_second = tl.maximum(max_val_second, max_val_first)
        
        # Accumulate channel result
        sum_val += max_val_second
    
    # Write final result
    output_offset = (n * stride_on + 0 * stride_oc + 
                    od * stride_od + oh * stride_oh + ow * stride_ow)
    tl.store(output_ptr + output_offset, sum_val, mask=mask)

def fused_pool_pool_sum_triton(x):
    batch, channels, in_d, in_h, in_w = x.shape
    
    # First pool: kernel=2, stride=2
    inter_d = (in_d - 2) // 2 + 1
    inter_h = (in_h - 2) // 2 + 1
    inter_w = (in_w - 2) // 2 + 1
    
    # Second pool: kernel=3, stride=3
    out_d = (inter_d - 3) // 3 + 1
    out_h = (inter_h - 3) // 3 + 1
    out_w = (inter_w - 3) // 3 + 1
    
    output = torch.empty((batch, 1, out_d, out_h, out_w), 
                         device=x.device, dtype=x.dtype)
    
    total_elements = batch * out_d * out_h * out_w
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_pool_pool_sum_kernel[grid](
        x, output,
        batch, channels, in_d, in_h, in_w,
        out_d, out_h, out_w,
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
        x = fused_pool_pool_sum_triton(x)
        return x
