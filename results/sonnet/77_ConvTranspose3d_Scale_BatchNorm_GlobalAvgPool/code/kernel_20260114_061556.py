import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_bn_gap_kernel_v2(
    input_ptr, output_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    batch_size, channels, spatial_size,
    stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Load batch norm parameters for this channel (cached in registers)
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Precompute normalization constants
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Combine scale_factor with gamma and inv_std for fewer ops in loop
    combined_scale = scale_factor * gamma * inv_std
    combined_bias = beta - mean * combined_scale
    
    # Base pointer for this (batch, channel)
    base_ptr = input_ptr + batch_idx * stride_b + channel_idx * stride_c
    
    # Accumulate sum over spatial dimensions using multiple blocks
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Process spatial elements in chunks
    for block_idx in range(NUM_BLOCKS):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < spatial_size
        
        # Load input values (coalesced access)
        x = tl.load(base_ptr + offs, mask=mask, other=0.0)
        
        # Fused scale + batch norm: x * combined_scale + combined_bias
        x = x * combined_scale + combined_bias
        
        # Accumulate for global average pooling
        acc += tl.where(mask, x, 0.0)
    
    # Sum reduction and compute average
    total_sum = tl.sum(acc, axis=0)
    avg = total_sum / spatial_size
    
    # Store result
    tl.store(output_ptr + batch_idx * channels + channel_idx, avg)


@triton.jit
def fused_scale_bn_gap_large_kernel(
    input_ptr, partial_sums_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    batch_size, channels, spatial_size,
    stride_b, stride_c,
    num_spatial_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (batch * channels, spatial_blocks)
    pid_bc = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    batch_idx = pid_bc // channels
    channel_idx = pid_bc % channels
    
    # Load batch norm parameters
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Precompute combined constants
    inv_std = 1.0 / tl.sqrt(var + eps)
    combined_scale = scale_factor * gamma * inv_std
    combined_bias = beta - mean * combined_scale
    
    # Base pointer for this (batch, channel)
    base_ptr = input_ptr + batch_idx * stride_b + channel_idx * stride_c
    
    # Compute offset for this spatial block
    start_idx = pid_s * BLOCK_SIZE
    offs = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offs < spatial_size
    
    # Load and process
    x = tl.load(base_ptr + offs, mask=mask, other=0.0)
    x = x * combined_scale + combined_bias
    x = tl.where(mask, x, 0.0)
    
    # Reduce within block
    block_sum = tl.sum(x, axis=0)
    
    # Store partial sum
    partial_idx = pid_bc * num_spatial_blocks + pid_s
    tl.store(partial_sums_ptr + partial_idx, block_sum)


@triton.jit
def reduce_partial_sums_kernel(
    partial_sums_ptr, output_ptr,
    batch_size, channels, num_spatial_blocks, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Load and sum partial sums for this (batch, channel)
    base_ptr = partial_sums_ptr + pid * num_spatial_blocks
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    num_full_blocks = num_spatial_blocks // BLOCK_SIZE
    remainder = num_spatial_blocks % BLOCK_SIZE
    
    for i in range(num_full_blocks):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(base_ptr + offs)
        acc += vals
    
    # Handle remainder
    if remainder > 0:
        offs = num_full_blocks * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_spatial_blocks
        vals = tl.load(base_ptr + offs, mask=mask, other=0.0)
        acc += vals
    
    total_sum = tl.sum(acc, axis=0)
    avg = total_sum / spatial_size
    
    tl.store(output_ptr + batch_idx * channels + channel_idx, avg)


def fused_scale_bn_gap(x, gamma, beta, running_mean, running_var, scale_factor, eps):
    batch_size, channels, D, H, W = x.shape
    spatial_size = D * H * W
    
    # Output shape: (batch_size, channels, 1, 1, 1)
    output = torch.empty((batch_size, channels, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    # Flatten spatial dimensions - ensure contiguous
    x_flat = x.view(batch_size, channels, spatial_size).contiguous()
    
    # Choose strategy based on spatial size
    BLOCK_SIZE = 1024
    
    if spatial_size <= 32768:  # Single kernel approach
        NUM_BLOCKS = triton.cdiv(spatial_size, BLOCK_SIZE)
        grid = (batch_size * channels,)
        
        fused_scale_bn_gap_kernel_v2[grid](
            x_flat, output,
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            batch_size, channels, spatial_size,
            x_flat.stride(0), x_flat.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BLOCKS=NUM_BLOCKS,
        )
    else:  # Two-pass approach for very large spatial sizes
        num_spatial_blocks = triton.cdiv(spatial_size, BLOCK_SIZE)
        partial_sums = torch.empty((batch_size * channels, num_spatial_blocks), 
                                   device=x.device, dtype=torch.float32)
        
        grid1 = (batch_size * channels, num_spatial_blocks)
        fused_scale_bn_gap_large_kernel[grid1](
            x_flat, partial_sums,
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            batch_size, channels, spatial_size,
            x_flat.stride(0), x_flat.stride(1),
            num_spatial_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        grid2 = (batch_size * channels,)
        REDUCE_BLOCK = min(128, triton.next_power_of_2(num_spatial_blocks))
        reduce_partial_sums_kernel[grid2](
            partial_sums, output,
            batch_size, channels, num_spatial_blocks, spatial_size,
            BLOCK_SIZE=REDUCE_BLOCK,
        )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        # Use cuDNN for transposed convolution (already optimized)
        x = self.conv_transpose(x)
        
        # Fused scale + batch norm + global average pooling
        x = fused_scale_bn_gap(
            x,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.scale_factor,
            self.eps
        )
        
        return x
