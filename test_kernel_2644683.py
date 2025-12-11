import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batch_norm_forward_kernel_2d(
    x_ptr,
    y_ptr,
    gamma_ptr,
    beta_ptr,
    running_mean_ptr,
    running_var_ptr,
    stride_xb,
    stride_xc,
    stride_xh,
    stride_xw,
    stride_yb,
    stride_yc,
    stride_yh,
    stride_yw,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """
    Optimized BatchNorm2d forward kernel that processes channels in parallel.
    Each block processes a group of channels and accumulates statistics across
    batches and spatial dimensions.
    """
    # Program indices
    pid_c = tl.program_id(0)  # Channel group index
    pid_hw = tl.program_id(1)  # Spatial tile index
    
    # Channel indices for this block
    c_idx = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_idx < C
    
    # Spatial indices for this block
    hw_idx = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    h = hw_idx // W
    w = hw_idx % W
    h_mask = h < H
    
    # Combine masks
    hw_mask = h_mask & (hw_idx < H * W)
    
    # Initialize accumulators
    sum_val = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    sum_sq_val = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # First pass: compute statistics
    for b in range(B):
        # Load spatial block for current batch and channel group
        x_ptrs = (
            x_ptr + 
            b * stride_xb + 
            c_idx[:, None] * stride_xc + 
            h[None, :] * stride_xh + 
            w[None, :] * stride_xw
        )
        mask = c_mask[:, None] & hw_mask[None, :]
        x_block = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Update accumulators
        sum_val += tl.sum(x_block, axis=1)
        sum_sq_val += tl.sum(x_block * x_block, axis=1)
        count += tl.sum(mask, axis=1).to(tl.float32)
    
    # Compute mean and variance for this channel group
    mean = sum_val / tl.maximum(count, 1.0)
    variance = (sum_sq_val / tl.maximum(count, 1.0)) - (mean * mean)
    
    # Load parameters and running stats for this channel group
    gamma = tl.load(gamma_ptr + c_idx, mask=c_mask, other=1.0)
    beta = tl.load(beta_ptr + c_idx, mask=c_mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + c_idx, mask=c_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_idx, mask=c_mask, other=1.0)
    
    # Update running statistics if in training mode
    if training:
        # Use Bessel's correction for unbiased variance estimate
        unbiased_var = variance * (count / tl.maximum(count - 1.0, 1.0))
        new_running_mean = running_mean * (1.0 - momentum) + mean * momentum
        new_running_var = running_var * (1.0 - momentum) + unbiased_var * momentum
        
        tl.store(running_mean_ptr + c_idx, new_running_mean, mask=c_mask)
        tl.store(running_var_ptr + c_idx, new_running_var, mask=c_mask)
        
        norm_mean = mean
        norm_var = variance
    else:
        norm_mean = running_mean
        norm_var = running_var
    
    # Compute normalization parameters
    inv_std = 1.0 / tl.sqrt(norm_var + eps)
    scale = gamma * inv_std
    shift = beta - gamma * norm_mean * inv_std
    
    # Second pass: apply normalization
    for b in range(B):
        # Load spatial block
        x_ptrs = (
            x_ptr + 
            b * stride_xb + 
            c_idx[:, None] * stride_xc + 
            h[None, :] * stride_xh + 
            w[None, :] * stride_xw
        )
        mask = c_mask[:, None] & hw_mask[None, :]
        x_block = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Apply normalization
        y_block = x_block * scale[:, None] + shift[:, None]
        
        # Store results
        y_ptrs = (
            y_ptr + 
            b * stride_yb + 
            c_idx[:, None] * stride_yc + 
            h[None, :] * stride_yh + 
            w[None, :] * stride_yw
        )
        tl.store(y_ptrs, y_block, mask=mask)


def triton_batch_norm_forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True
) -> torch.Tensor:
    """
    Optimized Triton implementation of BatchNorm2d forward pass.
    """
    B, C, H, W = x.shape
    y = torch.empty_like(x)
    
    # Ensure tensors are contiguous and on correct device
    x = x.contiguous()
    y = y.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    
    # Kernel configuration
    BLOCK_SIZE_C = min(32, C)  # Process up to 32 channels per block
    BLOCK_SIZE_HW = 64  # Process 64 spatial positions at once
    
    # Grid configuration
    grid_c = triton.cdiv(C, BLOCK_SIZE_C)
    total_hw = H * W
    grid_hw = triton.cdiv(total_hw, BLOCK_SIZE_HW)
    grid = (grid_c, grid_hw)
    
    # Launch kernel
    batch_norm_forward_kernel_2d[grid](
        x, y, gamma, beta, running_mean, running_var,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        B, C, H, W, eps, momentum, training,
        BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return y


class ModelNew(nn.Module):
    """
    High-performance BatchNorm2d implementation using optimized Triton kernels.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # For compatibility with PyTorch's BatchNorm
        self.track_running_stats = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, H, W = x.shape
        
        # Validate tensor sizes
        if num_channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels but got {num_channels}")
        
        # Move parameters to same device as input if needed
        if self.gamma.device != x.device:
            self.gamma.data = self.gamma.data.to(x.device)
            self.beta.data = self.beta.data.to(x.device)
            self.running_mean.data = self.running_mean.data.to(x.device)
            self.running_var.data = self.running_var.data.to(x.device)
        
        # Use optimized Triton kernel
        return triton_batch_norm_forward(
            x, self.gamma, self.beta,
            self.running_mean, self.running_var,
            self.eps, self.momentum, self.training
        )
