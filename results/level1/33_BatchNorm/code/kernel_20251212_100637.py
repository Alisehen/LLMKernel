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
    Compute sum and sum of squares per channel.
    Optimized for 2D parallelism: (C, N)
    """
    pid_c = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Channel offset
    channel_idx = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channel_mask = channel_idx < C
    
    # Initialize accumulators
    sum_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Strides for tensor indexing
    batch_stride = C * H * W
    channel_stride = H * W
    
    # Reduce over spatial dimensions for this batch
    base_idx = pid_n * batch_stride + channel_idx * channel_stride
    
    # Vectorized spatial reduction
    for spatial_start in range(0, H * W, BLOCK_SIZE):
        spatial_idx = spatial_start + tl.arange(0, BLOCK_SIZE)
        spatial_mask = spatial_idx < H * W
        
        # Load vectorized chunk
        offsets = base_idx[:, None] + spatial_idx[None, :]
        x_val = tl.load(x_ptr + offsets, 
                        mask=channel_mask[:, None] & spatial_mask[None, :], 
                        other=0.0)
        
        # Accumulate sums
        sum_acc += tl.sum(x_val, axis=1)
        sum_sq_acc += tl.sum(x_val * x_val, axis=1)
    
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Apply batch normalization with 3D grid for optimal parallelism.
    Uses vectorized loads/stores and optimized memory access patterns.
    """
    # 3D grid: channels × batches × spatial groups
    pid_c = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)
    
    # Channel offset with vectorization
    channel_idx = pid_c * BLOCK_M + tl.arange(0, BLOCK_M)
    channel_mask = channel_idx < C
    
    # Load normalization parameters once per channel group
    mean = tl.load(mean_ptr + channel_idx, mask=channel_mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=channel_mask, other=0.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_mask, other=0.0)
    
    # Precompute normalization factor with stability epsilon
    inv_std = tl.math.rsqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale
    
    # Base index for this batch
    batch_stride = C * H * W
    channel_stride = H * W
    base_offset = pid_b * batch_stride + channel_idx[:, None] * channel_stride
    
    # Process vectorized spatial chunks
    for i in range(BLOCK_N):
        # Current vector's start index in the spatial dimension
        vector_hw_start = pid_hw * BLOCK_N * VEC_SIZE + i * VEC_SIZE
        
        # Skip if this vector is completely out of bounds
        if vector_hw_start >= H * W:
            break
        
        # Compute vector indices and mask
        vector_hw_idx = vector_hw_start + tl.arange(0, VEC_SIZE)
        vector_mask = vector_hw_idx < H * W
        
        # Only process if at least one element is valid
        if tl.sum(vector_mask) > 0:
            # Compute offsets for this vector
            offsets = base_offset[:, None] + vector_hw_idx[None, :]
            
            # Vectorized load with masks
            x = tl.load(x_ptr + offsets, 
                        mask=channel_mask[:, None] & vector_mask[None, :], 
                        other=0.0)
            
            # Apply normalization (vectorized)
            y = x * scale[:, None] + shift[:, None]
            
            # Vectorized store
            tl.store(y_ptr + offsets, y,
                     mask=channel_mask[:, None] & vector_mask[None, :])


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
    Triton-optimized batch normalization forward pass with 3D grid.
    """
    N, C, H, W = x.shape
    
    # Output tensor
    y = torch.empty_like(x)
    
    if training:
        # Training mode: compute batch statistics
        sum_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        sum_sq_tensor = torch.zeros(C, device=x.device, dtype=torch.float32)
        
        # Compute sum and sum of squares with 2D grid
        BLOCK_SIZE_REDUCE = 128
        grid_reduce = (
            triton.cdiv(C, BLOCK_SIZE_REDUCE),
            N,
            1,
        )
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
        grid_update = (triton.cdiv(C, BLOCK_SIZE_UPDATE), 1, 1)
        batch_norm_update_kernel[grid_update](
            running_mean,
            running_var,
            mean_tensor,
            var_tensor,
            momentum,
            C,
            BLOCK_SIZE=BLOCK_SIZE_UPDATE,
        )
        
        apply_mean = mean_tensor
        apply_var = var_tensor
    else:
        # Inference mode: use running statistics
        apply_mean = running_mean
        apply_var = running_var
    
    # Apply normalization with 3D grid
    BLOCK_M = 32  # Channels per block
    BLOCK_N = 4   # Spatial groups per block
    VEC_SIZE = 16  # Vectorization factor
    
    # Calculate 3D grid: channels × batches × spatial groups
    grid_apply = (
        triton.cdiv(C, BLOCK_M),
        N,
        triton.cdiv(H * W, BLOCK_N * VEC_SIZE),
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        VEC_SIZE=VEC_SIZE,
    )
    
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized BatchNorm2d replacement with 3D grid parallelism.
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
