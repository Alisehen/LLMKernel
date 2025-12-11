import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def batch_norm_forward_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute channel index
    c_idx = tl.program_id(0)
    
    # Compute total elements per channel
    elements_per_channel = N * H * W
    
    # Initialize accumulators
    mean_acc = tl.zeros((1,), dtype=tl.float32)
    var_acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load parameters for this channel
    gamma = tl.load(gamma_ptr + c_idx)
    beta = tl.load(beta_ptr + c_idx)
    running_mean = tl.load(running_mean_ptr + c_idx)
    running_var = tl.load(running_var_ptr + c_idx)
    
    if training:
        # Compute mean and variance
        for block_start in range(0, elements_per_channel, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < elements_per_channel
            
            # Compute memory offset for this block
            channel_offset = c_idx * elements_per_channel
            block_offsets = channel_offset + offsets
            
            # Load data
            x = tl.load(x_ptr + block_offsets, mask=mask, other=0.0).to(tl.float32)
            
            # Accumulate sum
            sum_val = tl.sum(x, axis=0)
            mean_acc += sum_val
            
            # Accumulate squared sum for variance
            x_squared = x * x
            sum_squared = tl.sum(x_squared, axis=0)
            var_acc += sum_squared
        
        # Compute final mean and variance
        mean = mean_acc / elements_per_channel
        var = (var_acc / elements_per_channel) - (mean * mean)
        
        # Update running statistics
        new_running_mean = (1 - momentum) * running_mean + momentum * mean
        new_running_var = (1 - momentum) * running_var + momentum * var
        
        tl.store(running_mean_ptr + c_idx, new_running_mean)
        tl.store(running_var_ptr + c_idx, new_running_var)
    else:
        # Use running statistics
        mean = running_mean
        var = running_var
    
    # Compute normalization parameters
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize all elements in this channel
    for block_start in range(0, elements_per_channel, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements_per_channel
        
        # Compute memory offset for this block
        channel_offset = c_idx * elements_per_channel
        block_offsets = channel_offset + offsets
        
        # Load data
        x = tl.load(x_ptr + block_offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Apply batch normalization
        x_normalized = (x - mean) * inv_std
        output = gamma * x_normalized + beta
        
        # Store result
        tl.store(output_ptr + block_offsets, output, mask=mask)

@triton.jit
def batch_norm_forward_optimized_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
):
    # Compute channel and block indices
    pid = tl.program_id(0)
    c_start = pid * CHANNELS_PER_BLOCK
    
    # Compute total elements per channel
    elements_per_channel = N * H * W
    
    # Process multiple channels per block
    for local_c in range(CHANNELS_PER_BLOCK):
        c_idx = c_start + local_c
        
        # Only process if within channel bounds
        if c_idx < C:
            # Initialize accumulators
            mean_acc = tl.zeros((1,), dtype=tl.float32)
            var_acc = tl.zeros((1,), dtype=tl.float32)
            
            # Load parameters for this channel
            gamma = tl.load(gamma_ptr + c_idx)
            beta = tl.load(beta_ptr + c_idx)
            running_mean = tl.load(running_mean_ptr + c_idx)
            running_var = tl.load(running_var_ptr + c_idx)
            
            channel_offset = c_idx * elements_per_channel
            
            if training:
                # Compute mean and variance with vectorized operations
                for block_start in range(0, elements_per_channel, BLOCK_SIZE):
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < elements_per_channel
                    
                    block_offsets = channel_offset + offsets
                    x = tl.load(x_ptr + block_offsets, mask=mask, other=0.0).to(tl.float32)
                    
                    # Use vectorized accumulation
                    sum_val = tl.sum(x, axis=0)
                    mean_acc += sum_val
                    
                    x_squared = x * x
                    sum_squared = tl.sum(x_squared, axis=0)
                    var_acc += sum_squared
                
                # Compute final statistics
                mean = mean_acc / elements_per_channel
                var = (var_acc / elements_per_channel) - (mean * mean)
                
                # Update running statistics
                new_running_mean = (1 - momentum) * running_mean + momentum * mean
                new_running_var = (1 - momentum) * running_var + momentum * var
                
                tl.store(running_mean_ptr + c_idx, new_running_mean)
                tl.store(running_var_ptr + c_idx, new_running_var)
            else:
                mean = running_mean
                var = running_var
            
            # Compute normalization parameters
            inv_std = 1.0 / tl.sqrt(var + eps)
            
            # Normalize all elements in this channel
            for block_start in range(0, elements_per_channel, BLOCK_SIZE):
                offsets = block_start + tl.arange(0, BLOCK_SIZE)
                mask = offsets < elements_per_channel
                
                block_offsets = channel_offset + offsets
                x = tl.load(x_ptr + block_offsets, mask=mask, other=0.0).to(tl.float32)
                
                # Apply batch normalization with fused operations
                x_normalized = (x - mean) * inv_std
                output = gamma * x_normalized + beta
                
                tl.store(output_ptr + block_offsets, output, mask=mask)

def triton_batch_norm_forward(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5
) -> torch.Tensor:
    # Get dimensions
    N, C, H, W = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Choose kernel based on tensor size
    if H * W >= 256:  # Use optimized kernel for larger spatial dimensions
        BLOCK_SIZE = 1024
        CHANNELS_PER_BLOCK = 4
        grid = lambda meta: (triton.cdiv(C, CHANNELS_PER_BLOCK),)
        
        batch_norm_forward_optimized_kernel[grid](
            x, gamma, beta, None, None, output,
            running_mean, running_var,
            N, C, H, W,
            eps=eps, momentum=momentum, training=training,
            BLOCK_SIZE=BLOCK_SIZE, CHANNELS_PER_BLOCK=CHANNELS_PER_BLOCK
        )
    else:
        BLOCK_SIZE = 512
        grid = (C,)
        
        batch_norm_forward_kernel[grid](
            x, gamma, beta, None, None, output,
            running_mean, running_var,
            N, C, H, W,
            eps=eps, momentum=momentum, training=training,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Hyperparameters
        self.momentum = 0.1
        self.eps = 1e-5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_batch_norm_forward(
            x, self.gamma, self.beta, 
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )
