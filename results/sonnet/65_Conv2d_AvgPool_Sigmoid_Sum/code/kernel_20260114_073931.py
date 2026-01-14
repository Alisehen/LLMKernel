import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_conv_avgpool_sigmoid_sum_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_h, in_w, out_h, out_w,
    pool_h, pool_w, pool_kernel,
    kernel_size,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wb, stride_wc, stride_wh, stride_ww,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one (batch, out_channel) pair
    b = pid // out_channels
    oc = pid % out_channels
    
    if b >= batch or oc >= out_channels:
        return
    
    # Accumulator for the sum
    acc = 0.0
    
    # Iterate over pooled output spatial locations
    for ph in range(pool_h):
        for pw in range(pool_w):
            # Average pooling: compute average over pool_kernel x pool_kernel window
            pool_sum = 0.0
            pool_count = 0.0
            
            for kh in range(pool_kernel):
                for kw in range(pool_kernel):
                    conv_h = ph * pool_kernel + kh
                    conv_w = pw * pool_kernel + kw
                    
                    if conv_h < out_h and conv_w < out_w:
                        # Compute convolution at this location
                        conv_val = 0.0
                        
                        # Load bias
                        bias_val = tl.load(bias_ptr + oc)
                        conv_val += bias_val
                        
                        # Convolve over input channels and kernel
                        for ic in range(in_channels):
                            for kh_conv in range(kernel_size):
                                for kw_conv in range(kernel_size):
                                    in_h_idx = conv_h + kh_conv
                                    in_w_idx = conv_w + kw_conv
                                    
                                    if in_h_idx < in_h and in_w_idx < in_w:
                                        x_idx = b * stride_xb + ic * stride_xc + in_h_idx * stride_xh + in_w_idx * stride_xw
                                        w_idx = oc * stride_wb + ic * stride_wc + kh_conv * stride_wh + kw_conv * stride_ww
                                        
                                        x_val = tl.load(x_ptr + x_idx)
                                        w_val = tl.load(weight_ptr + w_idx)
                                        conv_val += x_val * w_val
                        
                        pool_sum += conv_val
                        pool_count += 1.0
            
            # Average pooling
            if pool_count > 0.0:
                avg_val = pool_sum / pool_count
                # Apply sigmoid
                sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
                acc += sigmoid_val
    
    # Store the sum for this (batch, out_channel)
    tl.store(output_ptr + b * out_channels + oc, acc)


@triton.jit
def conv2d_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels,
    in_h, in_w, out_h, out_w, kernel_size,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wb, stride_wc, stride_wh, stride_ww,
    stride_ob, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    total = batch * out_channels * out_h * out_w
    if pid >= total:
        return
    
    # Decompose pid
    b = pid // (out_channels * out_h * out_w)
    rem = pid % (out_channels * out_h * out_w)
    oc = rem // (out_h * out_w)
    rem2 = rem % (out_h * out_w)
    oh = rem2 // out_w
    ow = rem2 % out_w
    
    acc = 0.0
    bias_val = tl.load(bias_ptr + oc)
    acc += bias_val
    
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = oh + kh
                iw = ow + kw
                
                if ih < in_h and iw < in_w:
                    x_idx = b * stride_xb + ic * stride_xc + ih * stride_xh + iw * stride_xw
                    w_idx = oc * stride_wb + ic * stride_wc + kh * stride_wh + kw * stride_ww
                    
                    x_val = tl.load(x_ptr + x_idx)
                    w_val = tl.load(weight_ptr + w_idx)
                    acc += x_val * w_val
    
    out_idx = b * stride_ob + oc * stride_oc + oh * stride_oh + ow * stride_ow
    tl.store(output_ptr + out_idx, acc)


@triton.jit
def avgpool_sigmoid_sum_kernel(
    x_ptr, output_ptr,
    batch, channels, in_h, in_w, out_h, out_w, pool_kernel,
    stride_xb, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    b = pid // channels
    c = pid % channels
    
    if b >= batch or c >= channels:
        return
    
    acc = 0.0
    
    for oh in range(out_h):
        for ow in range(out_w):
            pool_sum = 0.0
            pool_count = 0.0
            
            for kh in range(pool_kernel):
                for kw in range(pool_kernel):
                    ih = oh * pool_kernel + kh
                    iw = ow * pool_kernel + kw
                    
                    if ih < in_h and iw < in_w:
                        idx = b * stride_xb + c * stride_xc + ih * stride_xh + iw * stride_xw
                        val = tl.load(x_ptr + idx)
                        pool_sum += val
                        pool_count += 1.0
            
            if pool_count > 0.0:
                avg_val = pool_sum / pool_count
                sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
                acc += sigmoid_val
    
    tl.store(output_ptr + b * channels + c, acc)


def fused_forward(x, weight, bias, pool_kernel_size):
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Conv output size
    conv_h = in_h - kernel_size + 1
    conv_w = in_w - kernel_size + 1
    
    # Pool output size
    pool_h = conv_h // pool_kernel_size
    pool_w = conv_w // pool_kernel_size
    
    # Intermediate conv output
    conv_out = torch.empty((batch, out_channels, conv_h, conv_w), device=x.device, dtype=x.dtype)
    
    # Conv kernel
    total_conv = batch * out_channels * conv_h * conv_w
    grid_conv = lambda META: (triton.cdiv(total_conv, 1),)
    conv2d_kernel[grid_conv](
        x, weight, bias, conv_out,
        batch, in_channels, out_channels,
        in_h, in_w, conv_h, conv_w, kernel_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        BLOCK_SIZE=1,
    )
    
    # AvgPool + Sigmoid + Sum
    output = torch.empty((batch, out_channels), device=x.device, dtype=x.dtype)
    grid_pool = lambda META: (batch * out_channels,)
    avgpool_sigmoid_sum_kernel[grid_pool](
        conv_out, output,
        batch, out_channels, conv_h, conv_w, pool_h, pool_w, pool_kernel_size,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        BLOCK_SIZE=1,
    )
    
    # Sum over channels to get final output
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
