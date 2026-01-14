import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_groupnorm_tanh_hardswish_residual_logsumexp_kernel(
    x_conv_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    N, C, HW, groups,
    eps,
    stride_n, stride_c, stride_hw,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    
    hw_idx = pid_hw
    
    # Initialize logsumexp accumulators
    max_val = -1e20
    sum_exp = 0.0
    
    # Process each group
    for pid_g in range(groups):
        C_per_group = C // groups
        group_start = pid_g * C_per_group
        
        # Compute mean and variance for this group
        mean_val = 0.0
        var_val = 0.0
        count = C_per_group * HW
        
        for c_offset in range(0, C_per_group, BLOCK_C):
            c_block = tl.arange(0, BLOCK_C)
            c_idx = group_start + c_offset + c_block
            c_mask = (c_offset + c_block) < C_per_group
            
            for hw_offset in range(0, HW, BLOCK_SIZE):
                hw_block = tl.arange(0, BLOCK_SIZE)
                hw_idx_inner = hw_offset + hw_block
                hw_mask = hw_idx_inner < HW
                
                mask = c_mask[:, None] & hw_mask[None, :]
                ptrs = x_conv_ptr + pid_n * stride_n + c_idx[:, None] * stride_c + hw_idx_inner[None, :] * stride_hw
                vals = tl.load(ptrs, mask=mask, other=0.0)
                mean_val += tl.sum(vals)
                var_val += tl.sum(vals * vals)
        
        mean_val = mean_val / count
        var_val = var_val / count - mean_val * mean_val
        rstd = 1.0 / tl.sqrt(var_val + eps)
        
        # Apply normalization, tanh, hardswish, residual for this hw_idx
        for c_offset in range(0, C_per_group, BLOCK_C):
            c_block = tl.arange(0, BLOCK_C)
            c_idx = group_start + c_offset + c_block
            c_mask = (c_offset + c_block) < C_per_group
            
            gamma = tl.load(gamma_ptr + c_idx, mask=c_mask, other=1.0)
            beta = tl.load(beta_ptr + c_idx, mask=c_mask, other=0.0)
            
            ptrs = x_conv_ptr + pid_n * stride_n + c_idx * stride_c + hw_idx * stride_hw
            x_conv = tl.load(ptrs, mask=c_mask, other=0.0)
            
            # Group normalization
            x_norm = (x_conv - mean_val) * rstd
            x_norm = x_norm * gamma + beta
            
            # Tanh using exp
            exp_2x = tl.exp(2.0 * x_norm)
            x_tanh = (exp_2x - 1.0) / (exp_2x + 1.0)
            
            # HardSwish: x * relu6(x + 3) / 6
            x_plus_3 = x_tanh + 3.0
            relu6_val = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
            x_hard_swish = x_tanh * relu6_val / 6.0
            
            # Residual addition
            x_res = x_conv + x_hard_swish
            
            # Update logsumexp accumulators
            # First pass: find max
            block_max = tl.max(tl.where(c_mask, x_res, -1e20))
            new_max = tl.maximum(max_val, block_max)
            
            # Adjust sum_exp for new max
            sum_exp = sum_exp * tl.exp(max_val - new_max)
            max_val = new_max
            
            # Add current exp values
            exp_vals = tl.exp(x_res - max_val)
            sum_exp += tl.sum(tl.where(c_mask, exp_vals, 0.0))
    
    # Compute final logsumexp
    result = max_val + tl.log(sum_exp)
    out_ptr = output_ptr + pid_n * HW + hw_idx
    tl.store(out_ptr, result)


def fused_groupnorm_tanh_hardswish_residual_logsumexp(x_conv, gamma, beta, groups, eps):
    N, C, H, W = x_conv.shape
    HW = H * W
    output = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)
    
    BLOCK_SIZE = 256
    BLOCK_C = 16
    
    grid = (N, HW)
    fused_groupnorm_tanh_hardswish_residual_logsumexp_kernel[grid](
        x_conv, output,
        gamma, beta,
        N, C, HW, groups,
        eps,
        x_conv.stride(0), x_conv.stride(1), 1,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_C=BLOCK_C,
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
        
        # Fused: Group Normalization + Tanh + HardSwish + Residual Addition + LogSumExp
        x_logsumexp = fused_groupnorm_tanh_hardswish_residual_logsumexp(
            x_conv, self.gamma, self.beta, self.groups, self.eps
        )
        
        return x_logsumexp
