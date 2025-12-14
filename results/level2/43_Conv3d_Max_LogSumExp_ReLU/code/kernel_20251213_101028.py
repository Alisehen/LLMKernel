import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'num_stages': 2}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=8),  # Original
        triton.Config({'num_stages': 4}, num_warps=8),
        triton.Config({'num_stages': 3}, num_warps=4),
        triton.Config({'num_stages': 3}, num_warps=16),
    ],
    key=['n_channel', 'num_spatial_positions'],
)
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
    num_stages: tl.constexpr,
):
    """
    Optimized kernel with 2D grid layout and increased work per thread block.
    Each thread block handles SPATIAL_BLOCK spatial positions and reduces over channels.
    """
    pid_batch = tl.program_id(0)
    pid_spatial_block = tl.program_id(1)
    
    num_spatial_positions = n_depth * n_height * n_width
    spatial_block_start = pid_spatial_block * SPATIAL_BLOCK
    
    # Thread IDs for channel dimension
    tid = tl.arange(0, BLOCK_SIZE)
    # Spatial offsets within block
    spatial_offsets = tl.arange(0, SPATIAL_BLOCK)
    spatial_indices = spatial_block_start + spatial_offsets
    valid_spatial_mask = spatial_indices < num_spatial_positions
    
    # Convert linear spatial index to 3D coordinates
    spatial_idx = tl.where(valid_spatial_mask, spatial_indices, 0)
    width_idx = spatial_idx % n_width
    height_idx = (spatial_idx // n_width) % n_height
    depth_idx = spatial_idx // (n_height * n_width)
    
    # Initialize max_vals and sum_exps for each spatial position
    max_vals = tl.full((SPATIAL_BLOCK,), float('-inf'), dtype=tl.float32)
    sum_exps = tl.zeros((SPATIAL_BLOCK,), dtype=tl.float32)
    
    # Pre-compute spatial dimensions to reduce repeated calculations
    height_width = n_height * n_width
    channel_spatial = n_channel * num_spatial_positions
    
    # First pass: compute max for each spatial position across channels
    for channel_block_start in range(0, n_channel, BLOCK_SIZE):
        channel_indices = channel_block_start + tid
        channel_mask = channel_indices < n_channel
        
        # Compute offsets efficiently with pre-computed values
        base_offset = pid_batch * channel_spatial + channel_block_start * num_spatial_positions
        spatial_base = depth_idx * height_width + height_idx * n_width + width_idx
        
        # Vectorized offset calculation
        offsets = base_offset + channel_indices[:, None] * num_spatial_positions + spatial_base[None, :]
        
        # Load values with proper masking
        values = tl.load(input_ptr + offsets, 
                        mask=channel_mask[:, None] & valid_spatial_mask[None, :],
                        other=float('-inf'))
        
        # Update max values - use max reduction across channels
        block_max = tl.max(values, axis=0)
        max_vals = tl.maximum(max_vals, block_max)
    
    # Second pass: compute sum of exponentials using global max
    for channel_block_start in range(0, n_channel, BLOCK_SIZE):
        channel_indices = channel_block_start + tid
        channel_mask = channel_indices < n_channel
        
        # Reuse pre-computed values
        base_offset = pid_batch * channel_spatial + channel_block_start * num_spatial_positions
        spatial_base = depth_idx * height_width + height_idx * n_width + width_idx
        
        offsets = base_offset + channel_indices[:, None] * num_spatial_positions + spatial_base[None, :]
        
        values = tl.load(input_ptr + offsets,
                        mask=channel_mask[:, None] & valid_spatial_mask[None, :],
                        other=0.0)
        
        # Compute exp(values - max_vals) with proper broadcasting
        exp_values = tl.exp(values - max_vals[None, :])
        # Mask out invalid channels before summing
        exp_values = tl.where(channel_mask[:, None] & valid_spatial_mask[None, :], exp_values, 0.0)
        block_sum = tl.sum(exp_values, axis=0)
        sum_exps += block_sum
    
    # Compute logsumexp and apply ReLU
    # Handle edge case where sum_exps might be 0 (log(0) = -inf)
    log_sum_exps = tl.log(tl.maximum(sum_exps, 1e-10)) + max_vals
    results = tl.maximum(log_sum_exps, 0.0)
    
    # Store results - reshape to proper output format
    output_offsets = (pid_batch * num_spatial_positions + spatial_indices)
    tl.store(output_ptr + output_offsets, results, mask=valid_spatial_mask)

def triton_logsumexp_relu_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper function with 2D grid layout and autotuning.
    """
    n_batch, n_channel, n_depth, n_height, n_width = x.shape
    num_spatial_positions = n_depth * n_height * n_width
    
    # Output shape: (batch_size, 1, depth, height, width)
    output = torch.empty(n_batch, 1, n_depth, n_height, n_width,
                        device=x.device, dtype=x.dtype)
    
    # Configuration optimized for RTX 4090 (Ada Lovelace)
    BLOCK_SIZE = 256  # Channel dimension block size
    SPATIAL_BLOCK = 4  # Spatial positions per thread block
    
    # Calculate 2D grid
    grid_batch = n_batch
    grid_spatial = triton.cdiv(num_spatial_positions, SPATIAL_BLOCK)
    
    # Flatten output for kernel storage
    output_flat = output.view(n_batch, -1)
    
    # Launch optimized kernel with autotuning
    logsumexp_relu_kernel_optimized[(grid_batch, grid_spatial)](
        x,
        output_flat,
        n_batch,
        n_channel,
        n_depth,
        n_height,
        n_width,
        BLOCK_SIZE=BLOCK_SIZE,
        SPATIAL_BLOCK=SPATIAL_BLOCK,
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
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width')
        """
        x = self.conv(x)
        x = self.max_pool(x)
        x = triton_logsumexp_relu_optimized(x)
        return x
