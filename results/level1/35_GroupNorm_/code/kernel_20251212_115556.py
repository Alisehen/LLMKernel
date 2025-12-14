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
    
    # Initialize accumulation as scalars
    sum_val = 0.0
    sum_sq_val = 0.0
    
    # Process elements in blocks
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        block_offset = block_start + block_idx
        mask = block_offset < group_size
        
        # Compute indices
        channel_in_group = block_offset // (H * W)
        spatial_idx = block_offset % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        # Load with vectorization if possible
        x_offsets = batch_offset + channel_offset + spatial_offset
        
        # Handle vectorized loads
        if VEC_SIZE > 1:
            # Create vectorized offsets
            vec_offsets = x_offsets[:, None] + tl.arange(0, VEC_SIZE)[None, :]
            vec_mask = mask[:, None] & (vec_offsets < (batch_offset + C * H * W))
            
            # Load vectorized
            x_vals = tl.load(x_ptr + vec_offsets, mask=vec_mask, other=0.0)
            
            # Accumulate
            vec_sum = tl.sum(x_vals, axis=1)
            vec_sum_sq = tl.sum(x_vals * x_vals, axis=1)
            
            sum_val += tl.sum(vec_sum * mask.to(tl.float32))
            sum_sq_val += tl.sum(vec_sum_sq * mask.to(tl.float32))
        else:
            # Scalar load
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x_vals)
            sum_sq_val += tl.sum(x_vals * x_vals)
    
    # Compute mean and variance (scalars)
    mean = sum_val / group_size
    var = (sum_sq_val / group_size) - (mean * mean)
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Store statistics (scalars)
    tl.store(mean_ptr + stats_idx, mean)
    tl.store(rstd_ptr + stats_idx, rstd)
    
    # Normalize and apply weight/bias
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        block_offset = block_start + block_idx
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
        
        if VEC_SIZE > 1:
            # Vectorized operations
            vec_offsets = x_offsets[:, None] + tl.arange(0, VEC_SIZE)[None, :]
            vec_mask = mask[:, None] & (vec_offsets < (batch_offset + C * H * W))
            
            x_vals = tl.load(x_ptr + vec_offsets, mask=vec_mask, other=0.0)
            
            # Normalize
            normalized = (x_vals - mean) * rstd
            
            # Apply weight/bias if provided
            if weight_ptr is not None:
                weight_vals = tl.load(
                    weight_ptr + channel_idx[:, None] + tl.arange(0, VEC_SIZE)[None, :],
                    mask=vec_mask, other=1.0
                )
                normalized = normalized * weight_vals
            
            if bias_ptr is not None:
                bias_vals = tl.load(
                    bias_ptr + channel_idx[:, None] + tl.arange(0, VEC_SIZE)[None, :],
                    mask=vec_mask, other=0.0
                )
                normalized = normalized + bias_vals
            
            tl.store(out_ptr + vec_offsets, normalized, mask=vec_mask)
        else:
            # Scalar operations
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            normalized = (x_vals - mean) * rstd
            
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
    
    # Initialize accumulators as scalars
    sum_dout = 0.0
    sum_dout_xhat = 0.0
    
    # Channel-wise gradients (only if weight/bias gradients needed)
    if dweight_ptr is not None:
        dweight_acc = tl.zeros([channels_per_group], dtype=tl.float32)
        dbias_acc = tl.zeros([channels_per_group], dtype=tl.float32)
    
    # First pass: compute sums
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        block_offset = block_start + block_idx
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
        
        if VEC_SIZE > 1:
            vec_offsets = offsets[:, None] + tl.arange(0, VEC_SIZE)[None, :]
            vec_mask = mask[:, None] & (vec_offsets < (batch_offset + C * H * W))
            
            x_vals = tl.load(x_ptr + vec_offsets, mask=vec_mask, other=0.0)
            dout_vals = tl.load(dout_ptr + vec_offsets, mask=vec_mask, other=0.0)
            
            x_hat = (x_vals - mean) * rstd
            
            vec_sum_dout = tl.sum(dout_vals, axis=1)
            vec_sum_dout_xhat = tl.sum(dout_vals * x_hat, axis=1)
            
            sum_dout += tl.sum(vec_sum_dout * mask.to(tl.float32))
            sum_dout_xhat += tl.sum(vec_sum_dout_xhat * mask.to(tl.float32))
            
            # Accumulate weight/bias gradients
            if dweight_ptr is not None:
                for c in range(channels_per_group):
                    channel_mask = mask & (channel_in_group == c)
                    if tl.sum(channel_mask) > 0:
                        channel_vec_mask = vec_mask & (channel_in_group[:, None] == c)
                        dweight_acc = dweight_acc.at(c).add(
                            tl.sum(dout_vals * x_hat * channel_vec_mask.to(tl.float32))
                        )
                        dbias_acc = dbias_acc.at(c).add(
                            tl.sum(dout_vals * channel_vec_mask.to(tl.float32))
                        )
        else:
            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            dout_vals = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
            
            x_hat = (x_vals - mean) * rstd
            
            sum_dout += tl.sum(dout_vals)
            sum_dout_xhat += tl.sum(dout_vals * x_hat)
            
            # Accumulate weight/bias gradients
            if dweight_ptr is not None:
                for c in range(channels_per_group):
                    channel_mask = mask & (channel_in_group == c)
                    if tl.sum(channel_mask) > 0:
                        dweight_acc = dweight_acc.at(c).add(
                            tl.sum(dout_vals * x_hat * channel_mask.to(tl.float32))
                        )
                        dbias_acc = dbias_acc.at(c).add(
                            tl.sum(dout_vals * channel_mask.to(tl.float32))
                        )
    
    # Compute scaling factors
    scale = rstd / group_size
    c1 = sum_dout_xhat * scale
    c2 = sum_dout * scale
    
    # Second pass: compute gradients
    for block_start in range(0, group_size, BLOCK_SIZE):
        block_idx = tl.arange(0, BLOCK_SIZE)
        block_offset = block_start + block_idx
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
        
        if VEC_SIZE > 1:
            vec_offsets = offsets[:, None] + tl.arange(0, VEC_SIZE)[None, :]
            vec_mask = mask[:, None] & (vec_offsets < (batch_offset + C * H * W))
            
            x_vals = tl.load(x_ptr + vec_offsets, mask=vec_mask, other=0.0)
            dout_vals = tl.load(dout_ptr + vec_offsets, mask=vec_mask, other=0.0)
            
            x_hat = (x_vals - mean) * rstd
            
            if weight_ptr is not None:
                weight_vals = tl.load(
                    weight_ptr + channel_idx[:, None] + tl.arange(0, VEC_SIZE)[None, :],
                    mask=vec_mask, other=1.0
                )
                dx_vals = weight_vals * rstd * (dout_vals - x_hat * c1 - c2)
            else:
                dx_vals = rstd * (dout_vals - x_hat * c1 - c2)
            
            tl.store(dx_ptr + vec_offsets, dx_vals, mask=vec_mask)
        else:
            x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            dout_vals = tl.load(dout_ptr + offsets, mask=mask, other=0.0)
            
            x_hat = (x_vals - mean) * rstd
            
            if weight_ptr is not None:
                weight_vals = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
                dx_vals = weight_vals * rstd * (dout_vals - x_hat * c1 - c2)
            else:
                dx_vals = rstd * (dout_vals - x_hat * c1 - c2)
            
            tl.store(dx_ptr + offsets, dx_vals, mask=mask)
    
    # Store weight/bias gradients
    if dweight_ptr is not None:
        for c in range(channels_per_group):
            channel_idx = channel_start + c
            if channel_idx < C:
                tl.atomic_add(dweight_ptr + channel_idx, dweight_acc[c])
                tl.atomic_add(dbias_ptr + channel_idx, dbias_acc[c])


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
    
    # Handle None pointers safely
    weight_ptr = weight if weight is not None else x
    bias_ptr = bias if bias is not None else x
    
    # Launch kernel
    group_norm_forward_kernel[grid](
        x, out, mean, rstd,
        weight_ptr,
        bias_ptr,
        N, C, H, W, G, eps,
        channels_per_group=channels_per_group,
        group_size=group_size
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
    
    # Create gradient tensors only if weight is provided
    if weight is not None:
        dweight = torch.zeros_like(weight)
        dbias = torch.zeros_like(weight)
    else:
        # Use dummy tensors
        dweight = x
        dbias = x
    
    # Grid: one program per batch*group combination
    grid = (N * G,)
    
    # Handle None pointers safely
    weight_ptr = weight if weight is not None else x
    dweight_ptr = dweight if weight is not None else x
    dbias_ptr = dbias if weight is not None else x
    
    # Launch kernel
    group_norm_backward_kernel[grid](
        dout, x, dx,
        dweight_ptr,
        dbias_ptr,
        mean, rstd,
        weight_ptr,
        N, C, H, W, G,
        channels_per_group=channels_per_group,
        group_size=group_size
    )
    
    return dx, dweight if weight is not None else None, dbias if weight is not None else None


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
