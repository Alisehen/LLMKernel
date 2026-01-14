import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_stats_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    N, C, D, H, W,
    groups,
    channels_per_group,
    spatial_size,
    elements_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute mean and variance for each group using parallel reduction."""
    pid = tl.program_id(0)
    batch_idx = pid // groups
    group_idx = pid % groups
    
    # Base offset for this batch and group
    base_channel = group_idx * channels_per_group
    batch_offset = batch_idx * C * spatial_size
    
    # Accumulate sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    
    # Total elements to process for this group
    total_group_elements = channels_per_group * spatial_size
    
    for block_start in range(0, total_group_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_group_elements
        
        # Convert linear offset within group to channel and spatial index
        c_local = offsets // spatial_size
        spatial_idx = offsets % spatial_size
        c = base_channel + c_local
        
        # Global index
        idx = batch_offset + c * spatial_size + spatial_idx
        
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(tl.where(mask, x, 0.0))
        sum_sq += tl.sum(tl.where(mask, x * x, 0.0))
    
    # Compute mean and variance
    mean = sum_val / elements_per_group
    var = sum_sq / elements_per_group - mean * mean
    
    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


@triton.jit
def fused_groupnorm_clamp_dropout_kernel(
    x_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    N, C, spatial_size,
    groups,
    channels_per_group,
    eps,
    min_value,
    max_value,
    dropout_p,
    scale,
    seed,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GroupNorm normalization + clamp + dropout."""
    pid = tl.program_id(0)
    
    total_elements = N * C * spatial_size
    
    # Shared offsets for all operations
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input using shared offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute indices from shared offsets
    n = offsets // (C * spatial_size)
    remainder = offsets % (C * spatial_size)
    c = remainder // spatial_size
    
    # Compute group index
    group_idx = c // channels_per_group
    
    # Load statistics
    stats_idx = n * groups + group_idx
    mean = tl.load(mean_ptr + stats_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + stats_idx, mask=mask, other=1.0)
    
    # Load gamma and beta (per channel)
    gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    inv_std = tl.rsqrt(var + eps)
    x_norm = (x - mean) * inv_std
    
    # Apply affine transform
    x_out = x_norm * gamma + beta
    
    # Clamp operation (min is redundant since clamp handles both bounds)
    x_out = tl.minimum(tl.maximum(x_out, min_value), max_value)
    
    # Dropout (only during training)
    if training:
        random = tl.rand(seed, offsets)
        dropout_mask = random > dropout_p
        x_out = tl.where(dropout_mask, x_out * scale, 0.0)
    
    # Store using shared offsets and mask
    tl.store(output_ptr + offsets, x_out, mask=mask)


def fused_groupnorm_min_clamp_dropout(x, gamma, beta, groups, eps, min_value, max_value, dropout_p, training):
    N, C, D, H, W = x.shape
    spatial_size = D * H * W
    channels_per_group = C // groups
    elements_per_group = channels_per_group * spatial_size
    
    output = torch.empty_like(x)
    
    # Allocate buffers for mean and variance
    mean = torch.empty(N * groups, device=x.device, dtype=x.dtype)
    var = torch.empty(N * groups, device=x.device, dtype=x.dtype)
    
    # First pass: compute statistics
    BLOCK_SIZE_STATS = 1024
    grid_stats = (N * groups,)
    
    group_norm_stats_kernel[grid_stats](
        x, mean, var,
        N, C, D, H, W,
        groups,
        channels_per_group,
        spatial_size,
        elements_per_group,
        BLOCK_SIZE=BLOCK_SIZE_STATS,
    )
    
    # Second pass: normalize and apply operations
    total_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item() if training else 0
    scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 1.0
    
    fused_groupnorm_clamp_dropout_kernel[grid](
        x, output,
        mean, var,
        gamma, beta,
        N, C, spatial_size,
        groups,
        channels_per_group,
        eps,
        min_value, max_value,
        dropout_p, scale,
        seed,
        training,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model with fused GroupNorm + min + clamp + dropout operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.groups = groups
        self.out_channels = out_channels
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p
        self.eps = 1e-5
        
        # GroupNorm parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x = self.conv(x)
        x = fused_groupnorm_min_clamp_dropout(
            x, self.gamma, self.beta, 
            self.groups, self.eps,
            self.min_value, self.max_value, 
            self.dropout_p, self.training
        )
        return x
