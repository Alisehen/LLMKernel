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
    Optimized reduction kernel for computing sum and sum of squares.
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
    
    # Process each batch and spatial position
    for b in range(N):
        for hw in range(total_spatial):
            base_idx = b * batch_stride + channel_idx * channel_stride + hw
            x_val = tl.load(x_ptr + base_idx, mask=channel_mask, other=0.0)
            sum_acc += x_val
            sum_sq_acc += x_val * x_val
    
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
):
    """
    Vectorized update kernel for running statistics.
    """
    pid = tl.program_id(axis=0)
    channel_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = channel_idx < C
    
    # Load current statistics
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=0.0)
    
    # Update with momentum
    new_running_mean = momentum * running_mean + (1.0 - momentum) * mean
    new_running_var = momentum * running_var + (1.0 - momentum) * var
    
    # Store updated statistics
    tl.store(running_mean_ptr + channel_idx, new_running_mean, mask=mask)
    tl.store(running_var_ptr + channel_idx, new_running_var, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 64, 'VEC_SIZE': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128, 'VEC_SIZE': 4}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 32, 'VEC_SIZE': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256, 'VEC_SIZE': 1}, num_stages=1, num_warps=4),
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
):
    """
    Optimized apply kernel with proper shared memory handling.
    """
    pid_c = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)
    
    # Channel indices for this block
    c_start = pid_c * BLOCK_C
    c_idx = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_idx < C
    
    # Load normalization parameters for this channel block
    mean_vals = tl.load(mean_ptr + c_idx, mask=c_mask, other=0.0)
    var_vals = tl.load(var_ptr + c_idx, mask=c_mask, other=0.0)
    weight_vals = tl.load(weight_ptr + c_idx, mask=c_mask, other=0.0)
    bias_vals = tl.load(bias_ptr + c_idx, mask=c_mask, other=0.0)
    
    # Compute normalization parameters
    inv_std = tl.math.rsqrt(var_vals + eps)
    scale = weight_vals * inv_std
    shift = bias_vals - mean_vals * scale
    
    # Spatial indices for this block
    hw_start = pid_hw * BLOCK_HW
    hw_idx = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < H * W
    
    # Precompute strides
    batch_stride = C * H * W
    channel_stride = H * W
    
    # Process each batch
    for b in range(N):
        batch_base = b * batch_stride
        
        # Prepare indices for vectorized load/store
        # We use broadcasting to combine channel and spatial dimensions
        c_offsets = c_idx[:, None] * channel_stride
        hw_offsets = hw_idx[None, :]
        
        # Compute input indices
        input_indices = batch_base + c_offsets + hw_offsets
        
        # Create combined mask for valid channels and spatial positions
        combined_mask = c_mask[:, None] & hw_mask[None, :]
        
        # Load input
        x_vals = tl.load(x_ptr + input_indices, mask=combined_mask, other=0.0)
        
        # Apply batch normalization
        y_vals = x_vals * scale[:, None] + shift[:, None]
        
        # Store output
        tl.store(y_ptr + input_indices, y_vals, mask=combined_mask)


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
        
        # Launch reduction kernel
        BLOCK_SIZE_REDUCE = 256
        grid_reduce = (triton.cdiv(C, BLOCK_SIZE_REDUCE),)
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
        
        # Launch update kernel
        BLOCK_SIZE_UPDATE = 256
        grid_update = (triton.cdiv(C, BLOCK_SIZE_UPDATE),)
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
    
    # Launch optimized apply kernel
    grid_apply = lambda meta: (
        triton.cdiv(C, meta['BLOCK_C']),
        triton.cdiv(H * W, meta['BLOCK_HW']),
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
        # Ensure tensors are contiguous
        x = x.contiguous()
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
