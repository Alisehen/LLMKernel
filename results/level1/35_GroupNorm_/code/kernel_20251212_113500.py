import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['group_size', 'N', 'G']
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
):
    batch_group = tl.program_id(0)
    batch_idx = batch_group // G
    group_idx = batch_group % G
    
    channel_start = group_idx * channels_per_group
    
    pid = tl.program_id(1)  # For parallel reduction across warps
    num_pids = tl.num_programs(1)
    
    sum_val = tl.zeros([BLOCK_SIZE // (32 * NUM_WARPS)], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE // (32 * NUM_WARPS)], dtype=tl.float32)
    
    # First warp computes the statistics
    if pid == 0:
        for g_offset in range(0, group_size, BLOCK_SIZE):
            elem_idx = g_offset + tl.arange(0, BLOCK_SIZE)
            mask = elem_idx < group_size
            
            channel_in_group = elem_idx // (H * W)
            spatial_idx = elem_idx % (H * W)
            h_idx = spatial_idx // W
            w_idx = spatial_idx % W
            
            channel_idx = channel_start + channel_in_group
            batch_offset = batch_idx * C * H * W
            channel_offset = channel_idx * H * W
            spatial_offset = h_idx * W + w_idx
            
            x_offset = batch_offset + channel_offset + spatial_offset
            x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            
            sum_val += x_val
            sum_sq += x_val * x_val
        
        total_sum = tl.sum(sum_val, axis=0)
        total_sum_sq = tl.sum(sum_sq, axis=0)
        
        mean = total_sum / group_size
        var = (total_sum_sq / group_size) - (mean * mean)
        rstd = 1.0 / tl.sqrt(var + eps)
        
        stats_idx = batch_idx * G + group_idx
        tl.store(mean_ptr + stats_idx, mean)
        tl.store(rstd_ptr + stats_idx, rstd)
    
    # All warps participate in normalization
    tl.debug_barrier()
    
    # Load statistics from first warp
    stats_idx = batch_idx * G + group_idx
    mean = tl.load(mean_ptr + stats_idx)
    rstd = tl.load(rstd_ptr + stats_idx)
    
    # Process normalization in parallel across warps
    warps_per_program = NUM_WARPS
    warp_id = pid
    elements_per_warp = (group_size + warps_per_program - 1) // warps_per_program
    warp_start = warp_id * elements_per_warp
    warp_end = tl.minimum(warp_start + elements_per_warp, group_size)
    
    for g_offset in range(warp_start, warp_end, BLOCK_SIZE):
        elem_idx = g_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < warp_end
        
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        x_offset = batch_offset + channel_offset + spatial_offset
        out_offset = x_offset
        
        x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
        normalized = (x_val - mean) * rstd
        
        if weight_ptr is not None and bias_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx, mask=mask)
            bias = tl.load(bias_ptr + channel_idx, mask=mask)
            out_val = normalized * weight + bias
        else:
            out_val = normalized
        
        tl.store(out_ptr + out_offset, out_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['group_size', 'N', 'G']
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
):
    batch_group = tl.program_id(0)
    batch_idx = batch_group // G
    group_idx = batch_group % G
    
    channel_start = group_idx * channels_per_group
    
    pid = tl.program_id(1)  # For parallel reduction
    num_pids = tl.num_programs(1)
    
    stats_idx = batch_idx * G + group_idx
    mean = tl.load(mean_ptr + stats_idx)
    rstd = tl.load(rstd_ptr + stats_idx)
    
    # Parallel reduction across warps
    local_sum1 = tl.zeros([BLOCK_SIZE // (32 * NUM_WARPS)], dtype=tl.float32)
    local_sum2 = tl.zeros([BLOCK_SIZE // (32 * NUM_WARPS)], dtype=tl.float32)
    
    warps_per_program = NUM_WARPS
    warp_id = pid
    elements_per_warp = (group_size + warps_per_program - 1) // warps_per_program
    warp_start = warp_id * elements_per_warp
    warp_end = tl.minimum(warp_start + elements_per_warp, group_size)
    
    for g_offset in range(warp_start, warp_end, BLOCK_SIZE):
        elem_idx = g_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < warp_end
        
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offset = batch_offset + channel_offset + spatial_offset
        
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        dout_val = tl.load(dout_ptr + offset, mask=mask, other=0.0)
        
        x_hat = (x_val - mean) * rstd
        
        local_sum1 += tl.where(mask, dout_val, 0.0)
        local_sum2 += tl.where(mask, dout_val * x_hat, 0.0)
    
    total_sum1 = tl.sum(local_sum1, axis=0)
    total_sum2 = tl.sum(local_sum2, axis=0)
    
    # Cross-warp reduction
    if NUM_WARPS > 1:
        shmem = tl.zeros([NUM_WARPS * 2], dtype=tl.float32)
        shmem_idx = warp_id * 2
        tl.store(shmem + shmem_idx, total_sum1)
        tl.store(shmem + shmem_idx + 1, total_sum2)
        tl.debug_barrier()
        
        if warp_id == 0:
            for i in range(1, NUM_WARPS):
                total_sum1 += tl.load(shmem + i * 2)
                total_sum2 += tl.load(shmem + i * 2 + 1)
    
    scale = rstd / group_size
    c1 = total_sum2 * scale
    c2 = total_sum1 * scale
    
    # Normalization pass
    for g_offset in range(warp_start, warp_end, BLOCK_SIZE):
        elem_idx = g_offset + tl.arange(0, BLOCK_SIZE)
        mask = elem_idx < warp_end
        
        channel_in_group = elem_idx // (H * W)
        spatial_idx = elem_idx % (H * W)
        h_idx = spatial_idx // W
        w_idx = spatial_idx % W
        
        channel_idx = channel_start + channel_in_group
        batch_offset = batch_idx * C * H * W
        channel_offset = channel_idx * H * W
        spatial_offset = h_idx * W + w_idx
        
        offset = batch_offset + channel_offset + spatial_offset
        
        x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        dout_val = tl.load(dout_ptr + offset, mask=mask, other=0.0)
        
        x_hat = (x_val - mean) * rstd
        
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + channel_idx, mask=mask)
            dx_val = weight * rstd * (dout_val - x_hat * c1 - c2)
        else:
            dx_val = rstd * (dout_val - x_hat * c1 - c2)
        
        tl.store(dx_ptr + offset, dx_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['group_size', 'N', 'G']
)
@triton.jit
def group_norm_backward_weight_bias_kernel(
    dout_ptr,
    x_ptr,
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
):
    channel_idx = tl.program_id(0)
    group_idx = channel_idx // channels_per_group
    
    pid = tl.program_id(1)  # Parallel reduction across batches
    num_pids = tl.num_programs(1)
    
    sum_dweight = 0.0
    sum_dbias = 0.0
    
    # Distribute batches across warps
    batches_per_warp = (N + NUM_WARPS - 1) // NUM_WARPS
    warp_start = pid * batches_per_warp
    warp_end = tl.minimum(warp_start + batches_per_warp, N)
    
    for batch_idx in range(warp_start, warp_end):
        stats_idx = batch_idx * G + group_idx
        mean = tl.load(mean_ptr + stats_idx)
        rstd = tl.load(rstd_ptr + stats_idx)
        
        for g_offset in range(0, group_size, BLOCK_SIZE):
            elem_idx = g_offset + tl.arange(0, BLOCK_SIZE)
            mask = elem_idx < group_size
            
            channel_in_group = elem_idx // (H * W)
            spatial_idx = elem_idx % (H * W)
            h_idx = spatial_idx // W
            w_idx = spatial_idx % W
            
            curr_channel = group_idx * channels_per_group + channel_in_group
            mask = mask & (curr_channel == channel_idx)
            
            if tl.sum(mask, axis=0) == 0:
                continue
            
            batch_offset = batch_idx * C * H * W
            channel_offset = channel_idx * H * W
            spatial_offset = h_idx * W + w_idx
            
            offset = batch_offset + channel_offset + spatial_offset
            
            x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            dout_val = tl.load(dout_ptr + offset, mask=mask, other=0.0)
            
            x_hat = (x_val - mean) * rstd
            
            if weight_ptr is not None:
                weight = tl.load(weight_ptr + channel_idx)
                sum_dweight += tl.sum(dout_val * x_hat * weight, axis=0)
                sum_dbias += tl.sum(dout_val * weight, axis=0)
            else:
                sum_dweight += tl.sum(dout_val * x_hat, axis=0)
                sum_dbias += tl.sum(dout_val, axis=0)
    
    # Atomic reduction across warps
    tl.atomic_add(dweight_ptr + channel_idx, sum_dweight)
    tl.atomic_add(dbias_ptr + channel_idx, sum_dbias)


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
    mean = torch.empty(N, G, dtype=x.dtype, device=x.device)
    rstd = torch.empty(N, G, dtype=x.dtype, device=x.device)
    
    grid = (N * G, 1)  # 2D grid for parallel reduction
    
    group_norm_forward_kernel[grid](
        x, out, mean, rstd, weight, bias,
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
    
    grid = (N * G, 1)  # 2D grid for parallel reduction
    
    group_norm_backward_kernel[grid](
        dout, x, dx, dweight, dbias, mean, rstd, weight,
        N, C, H, W, G,
        channels_per_group, group_size
    )
    
    if weight is not None:
        weight_grid = (C, 1)  # 2D grid for parallel batch processing
        group_norm_backward_weight_bias_kernel[weight_grid](
            dout, x, dweight, dbias, mean, rstd, weight,
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
        mean = torch.empty(x.size(0), self.num_groups, device=x.device)
        rstd = torch.empty(x.size(0), self.num_groups, device=x.device)
        
        dx, dweight, dbias = triton_group_norm_backward(
            dout, x, mean, rstd, self.weight, self.num_groups
        )
        return dx, dweight, dbias
