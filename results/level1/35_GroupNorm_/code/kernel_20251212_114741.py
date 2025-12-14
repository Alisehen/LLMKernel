import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'VEC_SIZE': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'VEC_SIZE': 2}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'VEC_SIZE': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'VEC_SIZE': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'VEC_SIZE': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'VEC_SIZE': 2}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 8, 'VEC_SIZE': 2}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 8, 'VEC_SIZE': 1}, num_stages=1),
    ],
    key=['group_size']
)
@triton.jit
def group_norm_forward_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    H,
    W,
    G,
    eps,
    channels_per_group: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    # Process one group per program - better for memory coalescing
    batch_group = tl.program_id(0)
    batch_idx = batch_group // G
    group_idx = batch_group % G
    channel_start = group_idx * channels_per_group
    
    # Pre-compute offsets for better instruction scheduling
    batch_offset = batch_idx * C * H * W
    stats_idx = batch_idx * G + group_idx
    
    # Vectorized accumulation buffers
    sum_acc = tl.zeros([BLOCK_SIZE // VEC_SIZE], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_SIZE // VEC_SIZE], dtype=tl.float32)
    
    # Process elements in vectorized chunks
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_offset = block_start + tl.arange(0, BLOCK_SIZE, VEC_SIZE)
        mask = block_offset < group_size
        
        # Pre-compute indices with vectorization
        channel_in_group = block_offset // (H * W)
        spatial_idx = block_offset % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        # Vectorized memory loads with proper cache hints
        x_offsets = batch_offset + channel_offset + spatial_offset
        
        # Load VEC_SIZE elements at once
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Vectorized accumulation
        sum_acc += tl.sum(x_vals, axis=1)
        sum_sq_acc += tl.sum(x_vals * x_vals, axis=1)
    
    # Reduce across threads in the block
    sum_val = tl.sum(sum_acc, axis=0)
    sum_sq = tl.sum(sum_sq_acc, axis=0)
    
    # Compute mean and variance
    mean = sum_val / group_size
    var = (sum_sq / group_size) - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store statistics
    tl.store(mean_ptr + stats_idx, mean)
    tl.store(rstd_ptr + stats_idx, rstd)
    
    # Normalize and apply weight/bias with vectorization
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_offset = block_start + tl.arange(0, BLOCK_SIZE, VEC_SIZE)
        mask = block_offset < group_size
        
        # Reuse pre-computed indices
        channel_in_group = block_offset // (H * W)
        spatial_idx = block_offset % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        x_offsets = batch_offset + channel_offset + spatial_offset
        out_offsets = x_offsets
        
        # Vectorized load
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        
        # Vectorized normalization
        normalized = (x_vals - mean) * rstd
        
        # Vectorized weight/bias application
        if weight_ptr is not None:
            weight_vals = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
            normalized = normalized * weight_vals
        
        if bias_ptr is not None:
            bias_vals = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
            normalized = normalized + bias_vals
        
        tl.store(out_ptr + out_offsets, normalized, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'VEC_SIZE': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'VEC_SIZE': 2}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'VEC_SIZE': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'VEC_SIZE': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'VEC_SIZE': 1}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 8, 'VEC_SIZE': 1}, num_stages=1),
    ],
    key=['group_size']
)
@triton.jit
def group_norm_backward_kernel(
    dout_ptr,
    x_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    mean_ptr,
    rstd_ptr,
    weight_ptr,
    N,
    C,
    H,
    W,
    G,
    channels_per_group: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    # Process one group per program
    batch_group = tl.program_id(0)
    batch_idx = batch_group // G
    group_idx = batch_group % G
    channel_start = group_idx * channels_per_group
    
    # Pre-compute offsets
    batch_offset = batch_idx * C * H * W
    stats_idx = batch_idx * G + group_idx
    
    # Load statistics
    mean = tl.load(mean_ptr + stats_idx)
    rstd = tl.load(rstd_ptr + stats_idx)
    
    # Vectorized accumulation buffers
    sum_dout_acc = tl.zeros([BLOCK_SIZE // VEC_SIZE], dtype=tl.float32)
    sum_dout_xhat_acc = tl.zeros([BLOCK_SIZE // VEC_SIZE], dtype=tl.float32)
    
    # Use shared memory for channel-wise accumulation
    sum_dweight = tl.zeros([channels_per_group], dtype=tl.float32)
    sum_dbias = tl.zeros([channels_per_group], dtype=tl.float32)
    
    # First pass: compute sums with vectorization
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_offset = block_start + tl.arange(0, BLOCK_SIZE, VEC_SIZE)
        mask = block_offset < group_size
        
        # Compute indices
        channel_in_group = block_offset // (H * W)
        spatial_idx = block_offset % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offsets = batch_offset + channel_offset + spatial_offset
        
        # Vectorized loads
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        dout_vals = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
        
        # Vectorized computation
        x_hat = (x_vals - mean) * rstd
        
        # Vectorized accumulation
        sum_dout_acc += tl.sum(dout_vals, axis=1)
        sum_dout_xhat_acc += tl.sum(dout_vals * x_hat, axis=1)
        
        # Accumulate weight/bias gradients with vector reduction
        if dweight_ptr is not None:
            for vec_idx in range(VEC_SIZE):
                vec_mask = mask & (block_offset + vec_idx < group_size)
                channel_mask = channel_in_group < channels_per_group
                active_mask = vec_mask & channel_mask
                
                if tl.sum(active_mask, axis=0) > 0:
                    dweight_vals = tl.where(active_mask, dout_vals[:, vec_idx] * x_hat[:, vec_idx], 0.0)
                    dbias_vals = tl.where(active_mask, dout_vals[:, vec_idx], 0.0)
                    
                    # Reduce across spatial dimensions
                    for c in tl.range(channels_per_group):
                        channel_active = active_mask & (channel_in_group == c)
                        if tl.sum(channel_active, axis=0) > 0:
                            sum_dweight = tl.sum(sum_dweight.at[c].add(tl.sum(dweight_vals * channel_active, axis=0)), axis=0)
                            sum_dbias = tl.sum(sum_dbias.at[c].add(tl.sum(dbias_vals * channel_active, axis=0)), axis=0)
    
    # Reduce across vector accumulators
    sum_dout = tl.sum(sum_dout_acc, axis=0)
    sum_dout_xhat = tl.sum(sum_dout_xhat_acc, axis=0)
    
    # Compute scaling factors
    scale = rstd / group_size
    c1 = sum_dout_xhat * scale
    c2 = sum_dout * scale
    
    # Second pass: compute gradients with vectorization
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_offset = block_start + tl.arange(0, BLOCK_SIZE, VEC_SIZE)
        mask = block_offset < group_size
        
        # Reuse indices
        channel_in_group = block_offset // (H * W)
        spatial_idx = block_offset % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offsets = batch_offset + channel_offset + spatial_offset
        
        # Vectorized loads
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        dout_vals = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
        
        # Vectorized normalized value
        x_hat = (x_vals - mean) * rstd
        
        # Vectorized gradient computation
        if weight_ptr is not None:
            weight_vals = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
            dx_vals = weight_vals * rstd * (dout_vals - x_hat * c1 - c2)
        else:
            dx_vals = rstd * (dout_vals - x_hat * c1 - c2)
        
        tl.store(dx_ptr + offsets, dx_vals, mask=mask)
    
    # Store weight/bias gradients with proper synchronization
    if dweight_ptr is not None:
        for c in range(channels_per_group):
            channel_idx = channel_start + c
            if channel_idx < C:
                tl.atomic_add(dweight_ptr + channel_idx, sum_dweight[c])
                tl.atomic_add(dbias_ptr + channel_idx, sum_dbias[c])


def triton_group_norm_forward(x, weight, bias, num_groups, eps=1e-5):
    N, C, *spatial_dims = x.shape
    if len(spatial_dims) == 1:
        H, W = spatial_dims[0], 1
    elif len(spatial_dims) == 2:
        H, W = spatial_dims
    else:
        H, W = 1, 1
        for dim in spatial_dims:
            H *= dim
    
    G = num_groups
    channels_per_group = C // G
    group_size = channels_per_group * H * W
    
    out = torch.empty_like(x)
    mean = torch.empty(N, G, dtype=torch.float32, device=x.device)
    rstd = torch.empty(N, G, dtype=torch.float32, device=x.device)
    
    # Grid: one program per batch*group combination
    grid = (N * G,)
    
    # Launch kernel
    group_norm_forward_kernel[grid](
        x, out, mean, rstd,
        weight if weight is not None else None,
        bias if bias is not None else None,
        N, C, H, W, G, eps,
        channels_per_group, group_size
    )
    
    return out, mean, rstd


def triton_group_norm_backward(dout, x, mean, rstd, weight, num_groups):
    N, C, *spatial_dims = x.shape
    if len(spatial_dims) == 1:
        H, W = spatial_dims[0], 1
    elif len(spatial_dims) == 2:
        H, W = spatial_dims
    else:
        H, W = 1, 1
        for dim in spatial_dims:
            H *= dim
    
    G = num_groups
    channels_per_group = C // G
    group_size = channels_per_group * H * W
    
    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight) if weight is not None else None
    dbias = torch.zeros_like(weight) if weight is not None else None
    
    # Grid: one program per batch*group combination
    grid = (N * G,)
    
    # Launch kernel
    group_norm_backward_kernel[grid](
        dout, x, dx,
        dweight if dweight is not None else None,
        dbias if dbias is not None else None,
        mean, rstd,
        weight if weight is not None else None,
        N, C, H, W, G,
        channels_per_group, group_size
    )
    
    return dx, dweight, dbias


class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = triton_group_norm_forward(
            x, self.weight, self.bias, self.num_groups, self.eps
        )
        return out
    
    def backward(self, dout: torch.Tensor, x: torch.Tensor) -> tuple:
        N, C, *spatial_dims = x.shape
        if len(spatial_dims) == 1:
            H, W = spatial_dims[0], 1
        elif len(spatial_dims) == 2:
            H, W = spatial_dims
        else:
            H, W = 1, 1
            for dim in spatial_dims:
                H *= dim
        
        G = self.num_groups
        channels_per_group = C // G
        
        # Compute statistics using PyTorch for backward pass
        x_flat = x.reshape(N, G, channels_per_group, H, W)
        x_flat = x_flat.reshape(N, G, -1)
        mean = x_flat.mean(dim=-1)
        var = x_flat.var(dim=-1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + self.eps)
        
        dx, dweight, dbias = triton_group_norm_backward(
            dout, x, mean, rstd, self.weight, self.num_groups
        )
        return dx, dweight, dbias
