import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_tanh_scale_sigmoid_kernel_2d(
    x_ptr,
    scaling_factor_ptr,
    bias_ptr,
    out_ptr,
    N,  # batch size
    C,  # number of channels
    spatial_size,  # D * H * W
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (channel_idx, spatial_block_idx)
    channel_idx = tl.program_id(0)
    spatial_block_idx = tl.program_id(1)
    
    # Load scaling_factor and bias once per channel (broadcast across spatial)
    scaling_factor = tl.load(scaling_factor_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Calculate spatial offsets within this block
    spatial_offsets = spatial_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total elements per channel across all batches
    total_spatial_per_channel = N * spatial_size
    
    # Mask for valid spatial positions
    mask = spatial_offsets < total_spatial_per_channel
    
    # Calculate batch index and spatial position within batch
    batch_idx = spatial_offsets // spatial_size
    spatial_pos = spatial_offsets % spatial_size
    
    # Calculate global offset: batch_idx * (C * spatial_size) + channel_idx * spatial_size + spatial_pos
    global_offsets = batch_idx * (C * spatial_size) + channel_idx * spatial_size + spatial_pos
    
    # Load input
    x = tl.load(x_ptr + global_offsets, mask=mask, other=0.0)
    
    # x = x * scaling_factor
    x = x * scaling_factor
    
    # x = tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * x)
    x = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # x = x * bias
    x = x * bias
    
    # x = sigmoid(x) = 1 / (1 + exp(-x))
    x = 1.0 / (1.0 + tl.exp(-x))
    
    # Store result
    tl.store(out_ptr + global_offsets, x, mask=mask)


def fused_scale_tanh_scale_sigmoid(x, scaling_factor, bias):
    # x shape: (N, C, D, H, W)
    # scaling_factor shape: (C, 1, 1, 1)
    # bias shape: (C, 1, 1, 1)
    
    N, C, D, H, W = x.shape
    spatial_size = D * H * W
    total_spatial_per_channel = N * spatial_size
    
    out = torch.empty_like(x)
    
    # Flatten scaling_factor and bias to 1D for easier indexing
    scaling_factor_flat = scaling_factor.view(-1).contiguous()
    bias_flat = bias.view(-1).contiguous()
    
    BLOCK_SIZE = 1024
    
    # 2D grid: (num_channels, num_spatial_blocks)
    num_spatial_blocks = triton.cdiv(total_spatial_per_channel, BLOCK_SIZE)
    grid = (C, num_spatial_blocks)
    
    fused_scale_tanh_scale_sigmoid_kernel_2d[grid](
        x,
        scaling_factor_flat,
        bias_flat,
        out,
        N,
        C,
        spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, then fuses scale, tanh, scale, sigmoid operations.
    Uses a 2D grid to eliminate expensive channel index computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_scale_tanh_scale_sigmoid(x, self.scaling_factor, self.bias)
        return x
