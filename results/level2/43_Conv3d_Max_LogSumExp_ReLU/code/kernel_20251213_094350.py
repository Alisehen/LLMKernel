import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def logsumexp_relu_kernel_optimized(
    input_ptr,
    output_ptr,
    n_batch,
    n_channel,
    n_depth,
    n_height,
    n_width,
    BLOCK_SIZE: tl.constexpr,
    SPATIAL_BLOCK: tl.constexpr,
):
    """
    Optimized kernel with 2D grid layout and increased work per thread block.
    Each thread block handles SPATIAL_BLOCK spatial positions and reduces over channels.
    """
    pid_batch = tl.program_id(0)
    pid_spatial_block = tl.program_id(1)
    
    num_spatial_positions = n_depth * n_height * n_width
    spatial_block_start = pid_spatial_block * SPATIAL_BLOCK
    
    # Thread IDs and masks
    tid = tl.arange(0, BLOCK_SIZE)
    spatial_offsets = tl.arange(0, SPATIAL_BLOCK)
    spatial_indices = spatial_block_start + spatial_offsets
    valid_spatial_mask = spatial_indices < num_spatial_positions
    
    # Convert spatial indices to 3D coordinates
    spatial_idx = tl.where(valid_spatial_mask, spatial_indices, 0)
    width_idx = spatial_idx % n_width
    height_idx = (spatial_idx // n_width) % n_height
    depth_idx = spatial_idx // (n_height * n_width)
    
    # Initialize per-thread max and sum_exp arrays
    max_vals = tl.full((SPATIAL_BLOCK,), -float('inf'), dtype=tl.float32)
    sum_exps = tl.zeros((SPATIAL_BLOCK,), dtype=tl.float32)
    
    # First pass: compute global max for each spatial position
    for channel_block_start in range(0, n_channel, BLOCK_SIZE):
        channel_indices = channel_block_start + tid
        channel_mask = channel_indices < n_channel
        
        # Compute offsets for all spatial positions in the block
        offsets = (pid_batch * n_channel * num_spatial_positions +
                  channel_indices[:, None] * num_spatial_positions +
                  depth_idx[None, :] * n_height * n_width +
                  height_idx[None, :] * n_width +
                  width_idx[None, :])
        
        # Load values with masking
        values = tl.load(input_ptr + offsets, 
                        mask=channel_mask[:, None] & valid_spatial_mask[None, :],
                        other=-float('inf'))
        
        # Update max values
        block_max = tl.max(values, axis=0)
        max_vals = tl.maximum(max_vals, block_max)
    
    # Second pass: compute sum of exponentials using global max
    for channel_block_start in range(0, n_channel, BLOCK_SIZE):
        channel_indices = channel_block_start + tid
        channel_mask = channel_indices < n_channel
        
        offsets = (pid_batch * n_channel * num_spatial_positions +
                  channel_indices[:, None] * num_spatial_positions +
                  depth_idx[None, :] * n_height * n_width +
                  height_idx[None, :] * n_width +
                  width_idx[None, :])
        
        values = tl.load(input_ptr + offsets,
                        mask=channel_mask[:, None] & valid_spatial_mask[None, :],
                        other=0.0)
        
        # Compute exp(values - max_vals) with proper broadcasting
        exp_values = tl.exp(values - max_vals[None, :])
        block_sum = tl.sum(exp_values, axis=0)
        sum_exps += block_sum
    
    # Compute logsumexp and apply ReLU
    log_sum_exps = tl.log(sum_exps) + max_vals
    results = tl.maximum(log_sum_exps, 0.0)
    
    # Store results
    output_offsets = (pid_batch * num_spatial_positions + spatial_indices)
    tl.store(output_ptr + output_offsets, results, mask=valid_spatial_mask)

def triton_logsumexp_relu_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper function with 2D grid layout.
    """
    n_batch, n_channel, n_depth, n_height, n_width = x.shape
    num_spatial_positions = n_depth * n_height * n_width
    
    output = torch.empty(n_batch, 1, n_depth, n_height, n_width,
                        device=x.device, dtype=x.dtype)
    
    # Configuration for Ada Lovelace (RTX 4090)
    BLOCK_SIZE = 256  # Optimized for channel reduction
    SPATIAL_BLOCK = 4  # Increased work per thread block
    
    # Calculate 2D grid
    grid_batch = n_batch
    grid_spatial = triton.cdiv(num_spatial_positions, SPATIAL_BLOCK)
    
    # Launch optimized kernel
    logsumexp_relu_kernel_optimized[(grid_batch, grid_spatial)](
        x,
        output,
        n_batch,
        n_channel,
        n_depth,
        n_height,
        n_width,
        BLOCK_SIZE=BLOCK_SIZE,
        SPATIAL_BLOCK=SPATIAL_BLOCK,
        num_warps=8,
        num_stages=3
    )
    
    return output

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, log sum exp, and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = triton_logsumexp_relu_optimized(x)
        return x
