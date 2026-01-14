import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_scale_bn_avgpool_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Load batch norm parameters for this channel (cached in registers)
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Precompute coefficients - fuse scale into BN
    inv_std = 1.0 / tl.sqrt(var + eps)
    coeff = scale_factor * gamma * inv_std
    bias = beta - mean * gamma * inv_std
    
    # Precompute constants
    HW = H * W
    
    # Base pointer for this (batch, channel)
    base_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    
    # Accumulate sum using register-based reduction
    acc = 0.0
    
    # Process in chunks of BLOCK_SIZE
    num_full_blocks = total_elements // BLOCK_SIZE
    remainder = total_elements % BLOCK_SIZE
    
    # Process full blocks
    for i in range(num_full_blocks):
        start = i * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        
        # Convert linear index to 3D indices
        d_idx = offs // HW
        hw_rem = offs % HW
        h_idx = hw_rem // W
        w_idx = hw_rem % W
        
        # Compute pointers
        ptrs = base_ptr + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        
        # Load and transform
        x_vals = tl.load(ptrs)
        y_vals = x_vals * coeff + bias
        
        # Accumulate
        acc += tl.sum(y_vals, axis=0)
    
    # Process remainder
    if remainder > 0:
        start = num_full_blocks * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements
        
        d_idx = offs // HW
        hw_rem = offs % HW
        h_idx = hw_rem // W
        w_idx = hw_rem % W
        
        ptrs = base_ptr + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x_vals = tl.load(ptrs, mask=mask, other=0.0)
        y_vals = x_vals * coeff + bias
        
        acc += tl.sum(tl.where(mask, y_vals, 0.0), axis=0)
    
    # Compute average and store
    avg = acc / total_elements
    tl.store(out_ptr + pid, avg)


@triton.jit
def fused_scale_bn_avgpool_kernel_simple(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    
    # Load batch norm parameters
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Precompute coefficients
    inv_std = 1.0 / tl.sqrt(var + eps)
    coeff = scale_factor * gamma * inv_std
    bias = beta - mean * gamma * inv_std
    
    HW = H * W
    base_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    
    acc = 0.0
    num_iters = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(num_iters):
        start = i * BLOCK_SIZE
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements
        
        d_idx = offs // HW
        hw_rem = offs % HW
        h_idx = hw_rem // W
        w_idx = hw_rem % W
        
        ptrs = base_ptr + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x_vals = tl.load(ptrs, mask=mask, other=0.0)
        y_vals = x_vals * coeff + bias
        
        acc += tl.sum(tl.where(mask, y_vals, 0.0), axis=0)
    
    avg = acc / total_elements
    tl.store(out_ptr + pid, avg)


def fused_scale_bn_avgpool(x, gamma, beta, running_mean, running_var, scale_factor, eps):
    N, C, D, H, W = x.shape
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    total_elements = D * H * W
    grid = (N * C,)
    
    # Use autotuned kernel
    fused_scale_bn_avgpool_kernel[grid](
        x, out.view(-1),
        gamma, beta, running_mean, running_var,
        scale_factor, eps,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        total_elements,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        
        if self.training:
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = torch.mean(x, dim=[2, 3, 4], keepdim=True)
            return x
        else:
            return fused_scale_bn_avgpool(
                x,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.scale_factor,
                self.eps
            )
