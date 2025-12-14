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
):
    """
    Compute sum and sum of squares per channel using atomic operations.
    Optimized for channel-parallel reduction with minimal memory traffic.
    """
    # Parallelize over channels and spatial positions
    pid = tl.program_id(axis=0)
    num_channels_per_block = tl.cdiv(C, BLOCK_SIZE)
    channel_group = pid // num_channels_per_block
    channel_idx = (pid % num_channels_per_block) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_idx < C
    
    # Initialize accumulators
    sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Strides for tensor indexing
    batch_stride = C * H * W
    channel_stride = H * W
    
    # Reduce over batch and spatial dimensions
    for b in range(N):
        for hw in range(H * W):
            # Compute base index
            base_idx = b * batch_stride + channel_idx * channel_stride + hw
            # Load with mask
            x_val = tl.load(x_ptr + base_idx, mask=channel_mask)
            # Accumulate
            sum_acc += x_val
            sum_sq_acc += x_val * x_val
    
    # Atomic add to global memory
    tl.atomic_add(sum_ptr + channel_idx, sum_acc, mask=channel_mask)
    tl.atomic_add(sum_sq_ptr + channel_idx, sum_sq_acc, mask=channel_mask)


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
    TRAINING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Apply batch normalization with optimal memory access patterns.
    Uses vectorized operations and precomputes normalization factors.
    """
    # 2D grid: channels × spatial positions
    pid_c = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)
    
    # Channel offset
    channel_offset = pid_c * BLOCK_SIZE
    channel_idx = channel_offset + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_idx < C
    
    # Spatial offset
    spatial_offset = pid_hw * BLOCK_SIZE
    spatial_idx = spatial_offset + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_idx < H * W
    
    # Load normalization parameters once per channel group
    if TRAINING:
        # Training mode: use computed statistics
        mean = tl.load(mean_ptr + channel_idx, mask=channel_mask, other=0.0)
        var = tl.load(var_ptr + channel_idx, mask=channel_mask, other=0.0)
    else:
        # Inference mode: use running statistics (passed in mean_ptr/var_ptr)
        mean = tl.load(mean_ptr + channel_idx, mask=channel_mask, other=0.0)
        var = tl.load(var_ptr + channel_idx, mask=channel_mask, other=0.0)
    
    weight = tl.load(weight_ptr + channel_idx, mask=channel_mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_mask, other=0.0)
    
    # Precompute normalization factor with stability epsilon
    inv_std = tl.math.rsqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale
    
    # Process batches
    batch_stride = C * H * W
    channel_stride = H * W
    
    for b in range(N):
        # Base index calculation
        base_idx = (b * batch_stride + 
                   channel_idx[:, None] * channel_stride + 
                   spatial_idx[None, :])
        
        # Vectorized load with masks
        x = tl.load(x_ptr + base_idx, 
                    mask=channel_mask[:, None] & spatial_mask[None, :], 
                    other=0.0)
        
        # Apply normalization (vectorized over channels × spatial)
        y = x * scale[:, None] + shift[:, None]
        
        # Store result
        tl.store(y_ptr + base_idx, y,
                 mask=channel_mask[:, None] & spatial_mask[None, :])


@triton.jit
def batch_norm_update_kernel(
    running_mean_ptr,
    running_var_ptr,
    mean_ptr,
    var_ptr,
    momentum: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Update running statistics with momentum.
    Optimized for parallel channel updates.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < C
    
    # Load current running statistics
    running_mean = tl.load(running_mean_ptr + offsets, mask=mask)
    running_var = tl.load(running_var_ptr + offsets, mask=mask)
    
    # Load batch statistics
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    
    # Update with momentum
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * var
    
    # Store back
    tl.store(running_mean_ptr + offsets, running_mean, mask=mask)
    tl.store(running_var_ptr + offsets, running_var, mask=mask)


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
    Triton-optimized batch normalization forward pass.
    """
    N, C, H, W = x.shape
    
    # Output tensor
    y = torch.empty_like(x)
    
    if training:
        # Training mode: compute batch statistics
        
        # Allocate temporary buffers
        sum_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        sum_sq_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        
        # Compute sum and sum of squares
        BLOCK_SIZE_REDUCE = 256
        grid_reduce = lambda meta: (triton.cdiv(C, meta['BLOCK_SIZE']) * C,)
        batch_norm_reduce_kernel[grid_reduce](
            x,
            sum_tensor,
            sum_sq_tensor,
            N, C, H, W,
            BLOCK_SIZE=BLOCK_SIZE_REDUCE,
        )
        
        # Compute mean and variance
        count = N * H * W
        mean_tensor = sum_tensor / count
        var_tensor = (sum_sq_tensor / count) - (mean_tensor * mean_tensor)
        
        # Update running statistics
        BLOCK_SIZE_UPDATE = 512
        grid_update = lambda meta: (triton.cdiv(C, meta['BLOCK_SIZE']),)
        batch_norm_update_kernel[grid_update](
            running_mean,
            running_var,
            mean_tensor,
            var_tensor,
            momentum,
            C,
            BLOCK_SIZE=BLOCK_SIZE_UPDATE,
        )
        
        # Apply normalization with batch statistics
        apply_mean = mean_tensor
        apply_var = var_tensor
    else:
        # Inference mode: use running statistics
        apply_mean = running_mean
        apply_var = running_var
    
    # Apply normalization
    BLOCK_SIZE_APPLY = 64
    grid_apply = lambda meta: (
        triton.cdiv(C, meta['BLOCK_SIZE']),
        triton.cdiv(H * W, meta['BLOCK_SIZE']),
    )
    batch_norm_apply_kernel[grid_apply](
        x,
        y,
        apply_mean,
        apply_var,
        weight,
        bias,
        eps,
        N, C, H, W,
        training,
        BLOCK_SIZE=BLOCK_SIZE_APPLY,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized BatchNorm2d replacement.
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
        # Ensure tensors are contiguous and on correct device
        x = x.contiguous()
        self.weight.data = self.weight.data.contiguous()
        self.bias.data = self.bias.data.contiguous()
        self.running_mean = self.running_mean.contiguous()
        self.running_var = self.running_var.contiguous()
        
        # Apply Triton batch normalization
        return triton_batch_norm2d(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.training,
            self.momentum,
            self.eps,
        )
