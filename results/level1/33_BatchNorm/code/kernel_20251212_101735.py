import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_reduce_kernel(
    x_ptr,
    sum_ptr,
    sum_sq_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Optimized reduction with improved memory access patterns.
    Uses software pipelining to hide memory latency.
    """
    pid = tl.program_id(axis=0)
    channel_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_idx < C
    
    # Initialize accumulators
    sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Precompute strides
    batch_stride = C * H * W
    channel_stride = H * W
    total_spatial = H * W
    
    # Process in stages for better memory latency hiding
    for b in range(N):
        for spatial_start in range(0, total_spatial, NUM_STAGES):
            # Preload multiple spatial positions
            x_vals = tl.zeros([BLOCK_SIZE, NUM_STAGES], dtype=tl.float32)
            
            # Load with software pipelining
            for stage in range(NUM_STAGES):
                hw = spatial_start + stage
                if hw < total_spatial:
                    base_idx = b * batch_stride + channel_idx * channel_stride + hw
                    x_val = tl.load(x_ptr + base_idx, mask=channel_mask, other=0.0)
                    x_vals[:, stage] = x_val
            
            # Process loaded values
            for stage in range(NUM_STAGES):
                if spatial_start + stage < total_spatial:
                    val = x_vals[:, stage]
                    sum_acc += val
                    sum_sq_acc += val * val
    
    # Store results
    tl.store(sum_ptr + channel_idx, sum_acc, mask=channel_mask)
    tl.store(sum_sq_ptr + channel_idx, sum_sq_acc, mask=channel_mask)


@triton.jit
def batch_norm_update_kernel(
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    var_ptr,
    momentum: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Vectorized update kernel for better memory throughput.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE * VEC_SIZE + tl.arange(0, BLOCK_SIZE * VEC_SIZE)
    mask = offsets < C
    
    # Vectorized loads and stores
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    
    # Update with momentum
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * var
    
    tl.store(running_mean_ptr + offsets, running_mean, mask=mask)
    tl.store(running_var_ptr + offsets, running_var, mask=mask)


# Optimized apply kernel with shared memory for parameters
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 64, 'VEC_SIZE': 4, 'NUM_STAGES': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128, 'VEC_SIZE': 4, 'NUM_STAGES': 3}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 32, 'VEC_SIZE': 2, 'NUM_STAGES': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256, 'VEC_SIZE': 1, 'NUM_STAGES': 1}, num_stages=1, num_warps=4),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def batch_norm_apply_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    eps: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Optimized apply kernel with:
    1. Shared memory for parameters to reduce global memory access
    2. Improved memory coalescing
    3. Better warp utilization
    """
    pid_c = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)
    
    # Shared memory for parameters
    shmem_mean = tl.zeros([BLOCK_C], dtype=tl.float32)
    shmem_var = tl.zeros([BLOCK_C], dtype=tl.float32)
    shmem_weight = tl.zeros([BLOCK_C], dtype=tl.float32)
    shmem_bias = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    # Load parameters into shared memory
    channel_offset = pid_c * BLOCK_C
    channel_idx = channel_offset + tl.arange(0, BLOCK_C)
    channel_mask = channel_idx < C
    
    if tl.program_id(axis=1) == 0:  # Only one thread per channel block loads params
        mean_vals = tl.load(mean_ptr + channel_idx, mask=channel_mask, other=0.0)
        var_vals = tl.load(var_ptr + channel_idx, mask=channel_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + channel_idx, mask=channel_mask, other=0.0)
        bias_vals = tl.load(bias_ptr + channel_idx, mask=channel_mask, other=0.0)
        
        # Store to shared memory
        tl.store(shmem_mean + tl.arange(0, BLOCK_C), mean_vals, mask=channel_mask)
        tl.store(shmem_var + tl.arange(0, BLOCK_C), var_vals, mask=channel_mask)
        tl.store(shmem_weight + tl.arange(0, BLOCK_C), weight_vals, mask=channel_mask)
        tl.store(shmem_bias + tl.arange(0, BLOCK_C), bias_vals, mask=channel_mask)
    
    tl.debug_barrier()  # Ensure all threads see the shared memory
    
    # Load from shared memory
    mean = tl.load(shmem_mean + tl.arange(0, BLOCK_C)[:, None])
    var = tl.load(shmem_var + tl.arange(0, BLOCK_C)[:, None])
    weight = tl.load(shmem_weight + tl.arange(0, BLOCK_C)[:, None])
    bias = tl.load(shmem_bias + tl.arange(0, BLOCK_C)[:, None])
    
    # Precompute normalization factors
    inv_std = tl.math.rsqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale
    
    # Process spatial positions with vectorization
    spatial_start = pid_hw * BLOCK_HW * VEC_SIZE
    spatial_idx = spatial_start + tl.arange(0, BLOCK_HW * VEC_SIZE)
    spatial_mask = spatial_idx < H * W
    
    # Strides
    batch_stride = C * H * W
    channel_stride = H * W
    
    # Process batches with software pipelining
    for b in range(N):
        batch_base = b * batch_stride + channel_offset * channel_stride
        
        # Vectorized loads and stores
        for vec_start in range(0, BLOCK_HW * VEC_SIZE, VEC_SIZE):
            vec_idx = vec_start + tl.arange(0, VEC_SIZE)
            vec_mask = spatial_mask & (vec_idx < BLOCK_HW * VEC_SIZE)
            
            if tl.reduce_and(vec_mask):  # All elements valid
                indices = batch_base + (channel_idx[:, None] * channel_stride) + (spatial_start + vec_idx)[None, :]
                x = tl.load(x_ptr + indices)
                y = x * scale + shift
                tl.store(y_ptr + indices, y)
            else:
                # Handle boundary with masking
                for i in range(VEC_SIZE):
                    if vec_start + i < BLOCK_HW * VEC_SIZE and spatial_mask[vec_start + i]:
                        hw_idx = spatial_start + vec_start + i
                        indices = batch_base + channel_idx * channel_stride + hw_idx
                        x = tl.load(x_ptr + indices, mask=channel_mask, other=0.0)
                        y = x * scale[:, 0] + shift[:, 0]
                        tl.store(y_ptr + indices, y, mask=channel_mask)


def triton_batch_norm2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Optimized batch normalization forward pass.
    """
    N, C, H, W = x.shape
    
    # Output tensor
    y = torch.empty_like(x)
    
    if training:
        # Training mode: compute batch statistics
        sum_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        sum_sq_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        
        # Optimized reduction with autotuned stages
        BLOCK_SIZE_REDUCE = 256
        NUM_STAGES_REDUCE = 3  # Increased for better latency hiding
        
        grid_reduce = (triton.cdiv(C, BLOCK_SIZE_REDUCE),)
        batch_norm_reduce_kernel[grid_reduce](
            x,
            sum_tensor,
            sum_sq_tensor,
            N, C, H, W,
            BLOCK_SIZE=BLOCK_SIZE_REDUCE,
            NUM_STAGES=NUM_STAGES_REDUCE,
        )
        
        # Compute mean and variance
        count = N * H * W
        mean_tensor = sum_tensor / count
        var_tensor = (sum_sq_tensor / count) - (mean_tensor * mean_tensor)
        
        # Vectorized update
        BLOCK_SIZE_UPDATE = 512
        VEC_SIZE_UPDATE = 4
        
        grid_update = (triton.cdiv(C, BLOCK_SIZE_UPDATE * VEC_SIZE_UPDATE),)
        batch_norm_update_kernel[grid_update](
            running_mean,
            running_var,
            mean_tensor,
            var_tensor,
            momentum,
            C,
            BLOCK_SIZE=BLOCK_SIZE_UPDATE,
            VEC_SIZE=VEC_SIZE_UPDATE,
        )
        
        apply_mean = mean_tensor
        apply_var = var_tensor
    else:
        # Inference mode: use running statistics
        apply_mean = running_mean
        apply_var = running_var
    
    # Launch optimized apply kernel
    grid_apply = (
        triton.cdiv(C, 128),  # Match BLOCK_C from autotune
        triton.cdiv(H * W, 64 * 4),  # Match BLOCK_HW * VEC_SIZE
    )
    
    batch_norm_apply_kernel[grid_apply](
        x, y, apply_mean, apply_var, weight, bias,
        eps, N, C, H, W,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Optimized Triton BatchNorm2d replacement.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # For compatibility with PyTorch's BatchNorm
        self.track_running_stats = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and in optimal layout
        x = x.contiguous(memory_format=torch.channels_last)
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()
        running_mean = self.running_mean.contiguous()
        running_var = self.running_var.contiguous()
        
        # Apply optimized batch normalization
        return triton_batch_norm2d(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            self.training,
            self.momentum,
            self.eps,
        )
