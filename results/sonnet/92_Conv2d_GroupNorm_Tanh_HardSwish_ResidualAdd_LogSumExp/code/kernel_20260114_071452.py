import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_groupnorm_tanh_hardswish_residual_kernel(
    x_conv_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    N, C, HW, groups,
    eps,
    stride_n, stride_c, stride_hw,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    
    # Calculate group parameters
    C_per_group = C // groups
    group_start = pid_g * C_per_group
    
    # Compute mean and variance for this group
    mean_val = tl.zeros((1,), dtype=tl.float32)
    var_val = tl.zeros((1,), dtype=tl.float32)
    count = C_per_group * HW
    
    for c_offset in range(0, C_per_group, BLOCK_C):
        c_block = tl.arange(0, BLOCK_C)
        c_idx = group_start + c_offset + c_block
        c_mask = c_block < (C_per_group - c_offset)
        
        for hw_offset in range(0, HW, BLOCK_SIZE):
            hw_block = tl.arange(0, BLOCK_SIZE)
            hw_idx = hw_offset + hw_block
            hw_mask = hw_block < (HW - hw_offset)
            
            mask = c_mask[:, None] & hw_mask[None, :]
            ptrs = x_conv_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx[None, :] * stride_hw
            vals = tl.load(ptrs, mask=mask, other=0.0)
            mean_val += tl.sum(vals)
            var_val += tl.sum(vals * vals)
    
    mean_val = mean_val / count
    var_val = var_val / count - mean_val * mean_val
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Apply normalization, tanh, hardswish, and residual
    for c_offset in range(0, C_per_group, BLOCK_C):
        c_block = tl.arange(0, BLOCK_C)
        c_idx = group_start + c_offset + c_block
        c_mask = c_block < (C_per_group - c_offset)
        
        gamma = tl.load(gamma_ptr + c_idx, mask=c_mask, other=1.0)
        beta = tl.load(beta_ptr + c_idx, mask=c_mask, other=0.0)
        
        for hw_offset in range(0, HW, BLOCK_SIZE):
            hw_block = tl.arange(0, BLOCK_SIZE)
            hw_idx = hw_offset + hw_block
            hw_mask = hw_block < (HW - hw_offset)
            
            mask = c_mask[:, None] & hw_mask[None, :]
            ptrs = x_conv_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx[None, :] * stride_hw
            x_val = tl.load(ptrs, mask=mask, other=0.0)
            
            # Group normalization
            x_norm = (x_val - mean_val) * rstd
            x_norm = x_norm * gamma[:, None] + beta[:, None]
            
            # Tanh
            exp_2x = tl.exp(2.0 * x_norm)
            x_tanh = (exp_2x - 1.0) / (exp_2x + 1.0)
            
            # HardSwish: x * relu6(x + 3) / 6
            x_plus_3 = x_tanh + 3.0
            relu6_val = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
            x_hardswish = x_tanh * relu6_val / 6.0
            
            # Residual addition
            x_res = x_val + x_hardswish
            
            out_ptrs = output_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx[None, :] * stride_hw
            tl.store(out_ptrs, x_res, mask=mask)


@triton.jit
def logsumexp_kernel(
    input_ptr, output_ptr,
    N, C, HW,
    stride_n, stride_c, stride_hw,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    hw_idx = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < HW
    
    # Find max for numerical stability - initialize as vector
    max_val = tl.full((BLOCK_HW,), -1e10, dtype=tl.float32)
    for c_offset in range(0, C, BLOCK_C):
        c_block = tl.arange(0, BLOCK_C)
        c_idx = c_offset + c_block
        c_mask = c_idx < C
        
        mask = c_mask[:, None] & hw_mask[None, :]
        ptrs = input_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx[None, :] * stride_hw
        vals = tl.load(ptrs, mask=mask, other=-1e10)
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))
    
    # Compute sum of exp(x - max)
    sum_exp = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for c_offset in range(0, C, BLOCK_C):
        c_block = tl.arange(0, BLOCK_C)
        c_idx = c_offset + c_block
        c_mask = c_idx < C
        
        mask = c_mask[:, None] & hw_mask[None, :]
        ptrs = input_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx[None, :] * stride_hw
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sum_exp += tl.sum(tl.exp(vals - max_val[None, :]), axis=0)
    
    # logsumexp = max + log(sum_exp)
    result = max_val + tl.log(sum_exp)
    
    out_ptrs = output_ptr + pid_n * stride_n + hw_idx * stride_hw
    tl.store(out_ptrs, result, mask=hw_mask)


def fused_operations(x_conv, gamma, beta, groups, eps):
    N, C, H, W = x_conv.shape
    HW = H * W
    output = torch.empty_like(x_conv)
    
    BLOCK_SIZE = 256
    BLOCK_C = 16
    
    grid = (N, groups)
    fused_groupnorm_tanh_hardswish_residual_kernel[grid](
        x_conv, output,
        gamma, beta,
        N, C, HW, groups, eps,
        x_conv.stride(0), x_conv.stride(1), x_conv.stride(2) * x_conv.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C=BLOCK_C,
    )
    
    return output


def logsumexp_triton(x):
    N, C, H, W = x.shape
    HW = H * W
    output = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 64
    BLOCK_HW = 256
    
    grid = (N, triton.cdiv(HW, BLOCK_HW))
    logsumexp_kernel[grid](
        x, output,
        N, C, HW,
        x.stride(0), x.stride(1), 1,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        
        # Fused: Group Normalization + Tanh + HardSwish + Residual Addition
        x_res = fused_operations(x_conv, self.gamma, self.beta, self.groups, self.eps)
        
        # LogSumExp
        x_logsumexp = logsumexp_triton(x_res)
        
        return x_logsumexp
