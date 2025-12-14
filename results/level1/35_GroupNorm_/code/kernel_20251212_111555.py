import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_forward_kernel(
    # Input tensor
    x_ptr,
    # Output tensor
    out_ptr,
    # Statistics
    mean_ptr,
    rstd_ptr,
    # Parameters
    weight_ptr,
    bias_ptr,
    # Tensor dimensions
    N,  # batch size
    C,  # channels
    H,  # height
    W,  # width
    G,  # number of groups
    eps,
    # Group properties
    channels_per_group: tl.constexpr,
    group_size: tl.constexpr,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Group Normalization forward kernel.
    
    Each program processes one group in one batch sample.
    """
    # Batch and group indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Compute group boundaries
    channel_start = group_idx * channels_per_group
    channel_end = channel_start + channels_per_group
    
    # Initialize accumulators for statistics
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process elements in the group
    for base_offset in range(0, group_size, BLOCK_SIZE):
        # Element indices within the group
        elem_idx = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < group_size
        
        # Convert 1D group index to 4D tensor indices
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        # Global channel index
        channel_idx = channel_start + channel_in_group
        
        # Compute memory offset
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        x_offset = batch_offset + channel_offset + spatial_offset
        
        # Load data
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Accumulate statistics
        sum_val += x_val
        sum_sq += x_val * x_val
    
    # Reduce across BLOCK_SIZE dimension
    total_sum = tl.sum(sum_val, axis=0)
    total_sum_sq = tl.sum(sum_sq, axis=0)
    
    # Compute mean and variance
    mean = total_sum / group_size
    var = (total_sum_sq / group_size) - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store statistics
    stats_idx = batch_idx * G + group_idx
    tl.store(mean_ptr + stats_idx, mean)
    tl.store(rstd_ptr + stats_idx, rstd)
    
    # Normalize and apply affine transformation
    for base_offset in range(0, group_size, BLOCK_SIZE):
        # Element indices within the group
        elem_idx = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < group_size
        
        # Convert 1D group index to 4D tensor indices
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        # Global channel index
        channel_idx = channel_start + channel_in_group
        
        # Compute memory offsets
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        x_offset = batch_offset + channel_offset + spatial_offset
        out_offset = x_offset  # same as input offset
        
        # Load data
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        
        # Normalize
        normalized = (x_val - mean) * rstd
        
        # Apply affine transformation if weights and biases are provided
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx)
            bias = tl.load(bias_ptr + channel_idx)
            out_val = normalized * weight + bias
        else:
            out_val = normalized
        
        # Store result
        tl.store(out_ptr + out_offset, out_val, mask=mask)


@triton.jit
def group_norm_backward_kernel(
    # Gradients and inputs
    dout_ptr,
    x_ptr,
    # Output gradients
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    # Statistics
    mean_ptr,
    rstd_ptr,
    # Parameters
    weight_ptr,
    # Tensor dimensions
    N,  # batch size
    C,  # channels
    H,  # height
    W,  # width
    G,  # number of groups
    # Group properties
    channels_per_group: tl.constexpr,
    group_size: tl.constexpr,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Group Normalization backward kernel.
    
    Each program processes one group in one batch sample.
    """
    # Batch and group indices
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)
    
    # Compute group boundaries
    channel_start = group_idx * channels_per_group
    channel_end = channel_start + channels_per_group
    
    # Load statistics
    stats_idx = batch_idx * G + group_idx
    mean = tl.load(mean_ptr + stats_idx)
    rstd = tl.load(rstd_ptr + stats_idx)
    
    # Initialize accumulators for gradient statistics
    sum1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_dweight = tl.zeros([channels_per_group], dtype=tl.float32)
    sum_dbias = tl.zeros([channels_per_group], dtype=tl.float32)
    
    # First pass: compute intermediate sums
    for base_offset in range(0, group_size, BLOCK_SIZE):
        # Element indices within the group
        elem_idx = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < group_size
        
        # Convert 1D group index to 4D tensor indices
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        # Global channel index
        channel_idx = channel_start + channel_in_group
        
        # Compute memory offsets
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offset = batch_offset + channel_offset + spatial_offset
        
        # Load data
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        dout_val = tl.load(dout_ptr + offset, mask=mask, other=0.0)
        
        # Compute normalized value
        x_hat = (x_val - mean) * rstd
        
        # Accumulate gradients for weight and bias
        if dweight_ptr is not None and dbias_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx)
            sum_dweight += tl.where(mask, dout_val * x_hat * weight, 0.0)
            sum_dbias += tl.where(mask, dout_val * weight, 0.0)
        
        # Accumulate intermediate sums for dx
        sum1 += tl.where(mask, dout_val, 0.0)
        sum2 += tl.where(mask, dout_val * x_hat, 0.0)
    
    # Reduce sums
    total_sum1 = tl.sum(sum1, axis=0)
    total_sum2 = tl.sum(sum2, axis=0)
    
    # Compute scale factor for dx
    scale = rstd / group_size
    c1 = total_sum2 * scale
    c2 = total_sum1 * scale
    
    # Second pass: compute dx
    for base_offset in range(0, group_size, BLOCK_SIZE):
        # Element indices within the group
        elem_idx = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < group_size
        
        # Convert 1D group index to 4D tensor indices
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        # Global channel index
        channel_idx = channel_start + channel_in_group
        
        # Compute memory offsets
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offset = batch_offset + channel_offset + spatial_offset
        
        # Load data
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        dout_val = tl.load(dout_ptr + offset, mask=mask, other=0.0)
        
        # Compute normalized value
        x_hat = (x_val - mean) * rstd
        
        # Compute dx
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx)
            dx_val = weight * rstd * (dout_val - x_hat * c1 - c2)
        else:
            dx_val = rstd * (dout_val - x_hat * c1 - c2)
        
        # Store dx
        tl.store(dx_ptr + offset, dx_val, mask=mask)
    
    # Store weight and bias gradients
    if dweight_ptr is not None and dbias_ptr is not None:
        total_dweight = tl.sum(sum_dweight, axis=0)
        total_dbias = tl.sum(sum_dbias, axis=0)
        
        # Atomic add to global gradient buffers
        for c in range(channels_per_group):
            channel_idx = channel_start + c
            tl.atomic_add(dweight_ptr + channel_idx, total_dweight)
            tl.atomic_add(dbias_ptr + channel_idx, total_dbias)


def triton_group_norm_forward(x, weight, bias, num_groups, eps=1e-5):
    """
    Triton implementation of Group Normalization forward pass.
    """
    # Get tensor dimensions
    N, C, *spatial_dims = x.shape
    if len(spatial_dims) == 1:
        H, W = spatial_dims[0], 1
    elif len(spatial_dims) == 2:
        H, W = spatial_dims
    else:
        # Handle higher dimensions by flattening
        H, W = 1, 1
        for dim in spatial_dims:
            H *= dim
    
    # Validate inputs
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"
    
    # Compute group properties
    G = num_groups
    channels_per_group = C // G
    group_size = channels_per_group * H * W
    
    # Allocate output and statistics
    out = torch.empty_like(x)
    mean = torch.empty(N, G, dtype=x.dtype, device=x.device)
    rstd = torch.empty(N, G, dtype=x.dtype, device=x.device)
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    
    # Launch grid: one program per batch per group
    grid = (N, G)
    
    # Launch kernel
    group_norm_forward_kernel[grid](
        x, out, mean, rstd, weight, bias,
        N, C, H, W, G, eps,
        channels_per_group, group_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out, mean, rstd


def triton_group_norm_backward(dout, x, mean, rstd, weight, num_groups):
    """
    Triton implementation of Group Normalization backward pass.
    """
    # Get tensor dimensions
    N, C, *spatial_dims = x.shape
    if len(spatial_dims) == 1:
        H, W = spatial_dims[0], 1
    elif len(spatial_dims) == 2:
        H, W = spatial_dims
    else:
        # Handle higher dimensions by flattening
        H, W = 1, 1
        for dim in spatial_dims:
            H *= dim
    
    # Compute group properties
    G = num_groups
    channels_per_group = C // G
    group_size = channels_per_group * H * W
    
    # Allocate gradients
    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight) if weight is not None else None
    dbias = torch.zeros_like(weight) if weight is not None else None
    
    # Configure kernel launch
    BLOCK_SIZE = 1024
    
    # Launch grid: one program per batch per group
    grid = (N, G)
    
    # Launch kernel
    group_norm_backward_kernel[grid](
        dout, x, dx, dweight, dbias, mean, rstd, weight,
        N, C, H, W, G,
        channels_per_group, group_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return dx, dweight, dbias


class ModelNew(nn.Module):
    """
    High-performance Group Normalization using Triton kernels.
    """
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Triton group normalization
        out, _, _ = triton_group_norm_forward(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
        return out
    
    def backward(self, dout: torch.Tensor, x: torch.Tensor) -> tuple:
        # For completeness, though normally called via autograd
        mean = torch.empty(x.size(0), self.num_groups, device=x.device)
        rstd = torch.empty(x.size(0), self.num_groups, device=x.device)
        
        dx, dweight, dbias = triton_group_norm_backward(
            dout, x, mean, rstd, self.weight, self.num_groups
        )
        
        return dx, dweight, dbias
