import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_avgpool_sigmoid_kernel(
    input_ptr, output_ptr,
    batch, channels, height, width,
    pool_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    out_height = height // pool_size
    out_width = width // pool_size
    total_elements = batch * channels * out_height * out_width
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    b = idx // (channels * out_height * out_width)
    rem = idx % (channels * out_height * out_width)
    c = rem // (out_height * out_width)
    rem2 = rem % (out_height * out_width)
    oh = rem2 // out_width
    ow = rem2 % out_width
    
    # Compute average pooling
    pool_sum = 0.0
    for ph in range(pool_size):
        for pw in range(pool_size):
            ih = oh * pool_size + ph
            iw = ow * pool_size + pw
            in_idx = b * (channels * height * width) + c * (height * width) + ih * width + iw
            valid = (b < batch) & (c < channels) & (ih < height) & (iw < width)
            val = tl.load(input_ptr + in_idx, mask=valid & mask, other=0.0)
            pool_sum += val
    
    avg_val = pool_sum / (pool_size * pool_size)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    tl.store(output_ptr + idx, sigmoid_val, mask=mask)


@triton.jit
def sum_reduction_kernel(
    input_ptr, output_ptr,
    batch, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid
    
    if b >= batch:
        return
    
    total = 0.0
    num_elements = channels * height * width
    
    for start_idx in range(0, num_elements, BLOCK_SIZE):
        idx = start_idx + tl.arange(0, BLOCK_SIZE)
        mask = idx < num_elements
        
        offset = b * num_elements + idx
        val = tl.load(input_ptr + offset, mask=mask, other=0.0)
        total += tl.sum(val)
    
    tl.store(output_ptr + b, total)


def fused_avgpool_sigmoid(x, pool_size):
    batch, channels, height, width = x.shape
    out_height = height // pool_size
    out_width = width // pool_size
    
    pooled = torch.empty((batch, channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    total_elements = batch * channels * out_height * out_width
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    fused_avgpool_sigmoid_kernel[grid](
        x, pooled,
        batch, channels, height, width,
        pool_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return pooled


def sum_all_dims(x):
    batch, channels, height, width = x.shape
    output = torch.empty((batch,), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    grid = (batch,)
    
    sum_reduction_kernel[grid](
        x, output,
        batch, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = fused_avgpool_sigmoid(x, self.pool_kernel_size)
        x = sum_all_dims(x)
        return x
