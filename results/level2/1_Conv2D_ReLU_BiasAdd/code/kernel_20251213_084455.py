import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_SPATIAL': 256, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 3}, num_warps=16),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 2}, num_warps=32),
        triton.Config({'BLOCK_SIZE_SPATIAL': 256, 'VECTOR_WIDTH': 2, 'NUM_STAGES': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512, 'VECTOR_WIDTH': 2, 'NUM_STAGES': 3}, num_warps=16),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024, 'VECTOR_WIDTH': 2, 'NUM_STAGES': 2}, num_warps=32),
    ],
    key=['spatial_size'],
)
@triton.jit
def bias_add_kernel_2d(
    input_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_bc = tl.program_id(axis=0)
    pid_spatial = tl.program_id(axis=1)
    
    batch_idx = pid_bc // channels
    channel_idx = pid_bc % channels
    
    bias = tl.load(bias_ptr + channel_idx)
    
    base_offset = (batch_idx * channels + channel_idx) * spatial_size
    input_block_ptr = input_ptr + base_offset
    output_block_ptr = output_ptr + base_offset
    
    spatial_offsets_vec = pid_spatial * (BLOCK_SIZE_SPATIAL * VECTOR_WIDTH) + tl.arange(0, BLOCK_SIZE_SPATIAL * VECTOR_WIDTH)
    spatial_mask_vec = spatial_offsets_vec < spatial_size
    
    input_vals = tl.load(input_block_ptr + spatial_offsets_vec, mask=spatial_mask_vec, other=0.0, cache_modifier=".cg")
    output_vals = input_vals + bias
    tl.store(output_block_ptr + spatial_offsets_vec, output_vals, mask=spatial_mask_vec, cache_modifier=".cg")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_SPATIAL': 256, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 3}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 2}, num_warps=16),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 1}, num_warps=32),
        triton.Config({'BLOCK_SIZE_SPATIAL': 256, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 3}, num_warps=16),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024, 'VECTOR_WIDTH': 4, 'NUM_STAGES': 2}, num_warps=32),
    ],
    key=['spatial_size'],
)
@triton.jit
def bias_add_kernel_2d_small(
    input_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
    VECTOR_WIDTH: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_bc = tl.program_id(axis=0)
    pid_spatial = tl.program_id(axis=1)
    
    batch_idx = pid_bc // channels
    channel_idx = pid_bc % channels
    
    bias = tl.load(bias_ptr + channel_idx)
    
    spatial_offsets = pid_spatial * BLOCK_SIZE_SPATIAL * VECTOR_WIDTH + tl.arange(0, BLOCK_SIZE_SPATIAL * VECTOR_WIDTH)
    spatial_mask = spatial_offsets < spatial_size
    
    base_offset = (batch_idx * channels + channel_idx) * spatial_size
    input_block_ptr = input_ptr + base_offset
    output_block_ptr = output_ptr + base_offset
    
    input_vals = tl.load(input_block_ptr + spatial_offsets, mask=spatial_mask, other=0.0, cache_modifier=".cg")
    output_vals = input_vals + bias
    tl.store(output_block_ptr + spatial_offsets, output_vals, mask=spatial_mask, cache_modifier=".cg")


def triton_bias_add_optimized(input_tensor: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    input_tensor = input_tensor.contiguous()
    bias = bias.contiguous()
    
    batch_size, channels, height, width = input_tensor.shape
    spatial_size = height * width
    
    bias_1d = bias.view(channels)
    output = torch.empty_like(input_tensor)
    
    if spatial_size >= 8192:
        grid = lambda META: (
            batch_size * channels,
            triton.cdiv(spatial_size, META['BLOCK_SIZE_SPATIAL'] * META['VECTOR_WIDTH']),
        )
        bias_add_kernel_2d[grid](
            input_tensor,
            bias_1d,
            output,
            batch_size,
            channels,
            spatial_size,
        )
    else:
        grid = lambda META: (
            batch_size * channels,
            triton.cdiv(spatial_size, META['BLOCK_SIZE_SPATIAL'] * META['VECTOR_WIDTH']),
        )
        bias_add_kernel_2d_small[grid](
            input_tensor,
            bias_1d,
            output,
            batch_size,
            channels,
            spatial_size,
        )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = triton_bias_add_optimized(x, self.bias)
        return x
