import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_mish_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Mish activation: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    # For numerical stability, use: softplus(x) = x if x > 20 else log(1 + exp(x))
    softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    
    # tanh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
    exp_2x = tl.exp(2.0 * softplus)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    out = x * tanh_val
    
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def batchnorm_forward_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr,
    mean_ptr, var_ptr,
    N, C, HW,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    offs_hw = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_hw = offs_hw < HW
    
    if training == 1:
        # Load mean and var computed for this channel
        mean = tl.load(mean_ptr + pid_c)
        var = tl.load(var_ptr + pid_c)
    else:
        # Use running statistics
        mean = tl.load(running_mean_ptr + pid_c)
        var = tl.load(running_var_ptr + pid_c)
    
    weight = tl.load(weight_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)
    
    # Process all N samples for this channel and HW block
    for n in range(N):
        idx = n * C * HW + pid_c * HW + offs_hw
        x = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
        
        # Normalize
        x_norm = (x - mean) / tl.sqrt(var + eps)
        
        # Scale and shift
        out = weight * x_norm + bias
        
        tl.store(out_ptr + idx, out, mask=mask_hw)

def fused_mish(x):
    N = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    fused_mish_kernel[grid](
        x, out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def batchnorm2d_triton(x, weight, bias, running_mean, running_var, training, momentum, eps):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    
    if training:
        # Compute mean and variance
        x_reshaped = x.view(N, C, HW)
        mean = x_reshaped.mean(dim=[0, 2])
        var = x_reshaped.var(dim=[0, 2], unbiased=False)
        
        # Update running statistics
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean * momentum)
            running_var.mul_(1 - momentum).add_(var * momentum)
        
        mean_ptr = mean.contiguous()
        var_ptr = var.contiguous()
    else:
        mean_ptr = running_mean
        var_ptr = running_var
    
    BLOCK_SIZE = 256
    grid = (C, triton.cdiv(HW, BLOCK_SIZE))
    
    batchnorm_forward_kernel[grid](
        x, out, weight, bias,
        running_mean, running_var,
        mean_ptr, var_ptr,
        N, C, HW,
        eps=eps,
        momentum=momentum,
        training=1 if training else 0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = self.conv(x)
        x = fused_mish(x)
        x = batchnorm2d_triton(
            x, self.bn_weight, self.bn_bias,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )
        return x
