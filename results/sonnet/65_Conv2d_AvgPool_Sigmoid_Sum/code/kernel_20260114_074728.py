import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, in_h, in_w,
    out_channels, out_h, out_w, kernel_size,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wb, stride_wc, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Total output elements
    total_out = batch * out_channels * out_h * out_w
    
    # Compute indices
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_out
    
    b = idx // (out_channels * out_h * out_w)
    rem = idx % (out_channels * out_h * out_w)
    oc = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w
    
    # Compute convolution
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = oh + kh
                iw = ow + kw
                
                # Load input
                x_idx = b * stride_xb + ic * stride_xc + ih * stride_xh + iw * stride_xw
                x_val = tl.load(x_ptr + x_idx, mask=mask & (ih < in_h) & (iw < in_w), other=0.0)
                
                # Load weight
                w_idx = oc * stride_wb + ic * stride_wc + kh * stride_wh + kw * stride_ww
                w_val = tl.load(weight_ptr + w_idx, mask=mask, other=0.0)
                
                acc += x_val * w_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    result = acc + bias_val
    
    # Store output
    out_idx = b * stride_ob + oc * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(output_ptr + out_idx, result, mask=mask)


@triton.jit
def fused_avgpool_sigmoid_sum_kernel(
    x_ptr, output_ptr,
    batch, out_channels, conv_h, conv_w,
    pool_kernel,
    stride_xb, stride_xc, stride_xh, stride_xw,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    b = pid_b
    c_start = pid_c * BLOCK_C
    c_offs = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offs < out_channels
    
    pool_h = conv_h // pool_kernel
    pool_w = conv_w // pool_kernel
    
    # Accumulator for sum across spatial dimensions
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Iterate over pooled spatial locations
    for ph in range(pool_h):
        for pw in range(pool_w):
            pool_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
            pool_count = 0.0
            
            # Average pooling window
            for kh in range(pool_kernel):
                for kw in range(pool_kernel):
                    conv_h_idx = ph * pool_kernel + kh
                    conv_w_idx = pw * pool_kernel + kw
                    
                    valid = (conv_h_idx < conv_h) & (conv_w_idx < conv_w)
                    
                    x_idx = b * stride_xb + c_offs * stride_xc + conv_h_idx * stride_xh + conv_w_idx * stride_xw
                    x_val = tl.load(x_ptr + x_idx, mask=c_mask & valid, other=0.0)
                    
                    pool_sum = tl.where(valid, pool_sum + x_val, pool_sum)
                    pool_count = tl.where(valid, pool_count + 1.0, pool_count)
            
            # Average and sigmoid
            avg_val = tl.where(pool_count > 0.0, pool_sum / pool_count, 0.0)
            sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
            acc += sigmoid_val
    
    # Store result
    out_idx = b * out_channels + c_offs
    tl.store(output_ptr + out_idx, acc, mask=c_mask)


def optimized_conv2d(x, weight, bias):
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    out_h = in_h - kernel_size + 1
    out_w = in_w - kernel_size + 1
    
    output = torch.empty((batch, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    
    total_out = batch * out_channels * out_h * out_w
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_out, BLOCK_SIZE),)
    
    fused_conv2d_kernel[grid](
        x, weight, bias, output,
        batch, in_channels, in_h, in_w,
        out_channels, out_h, out_w, kernel_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def fused_forward(x, weight, bias, pool_kernel_size):
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Convolution
    conv_out = optimized_conv2d(x, weight, bias)
    
    _, _, conv_h, conv_w = conv_out.shape
    
    # Fused avgpool + sigmoid + sum
    output = torch.empty((batch, out_channels), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 64
    grid_pool = (batch, triton.cdiv(out_channels, BLOCK_C))
    fused_avgpool_sigmoid_sum_kernel[grid_pool](
        conv_out, output,
        batch, out_channels, conv_h, conv_w,
        pool_kernel_size,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        BLOCK_C=BLOCK_C,
    )
    
    # Sum over channels
    result = output.sum(dim=1)
    return result


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        return fused_forward(x, self.weight, self.bias, self.pool_kernel_size)
