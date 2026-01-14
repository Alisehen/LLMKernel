import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_groupnorm_tanh_hardswish_residual_kernel(
    x_conv_ptr, output_ptr,
    gamma_ptr, beta_ptr,
    N, C, HW, groups,
    eps: tl.constexpr,
    stride_n, stride_c, stride_hw,
    BLOCK_HW: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    group_start = pid_g * C_PER_GROUP
    hw_offset = pid_hw * BLOCK_HW
    hw_idx = hw_offset + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < HW
    
    # Compute mean and variance for this group across all spatial locations
    mean_val = 0.0
    var_val = 0.0
    count = C_PER_GROUP * HW
    
    # Accumulate statistics
    for c in range(C_PER_GROUP):
        c_idx = group_start + c
        ptrs = x_conv_ptr + pid_n * stride_n + c_idx * stride_c + hw_idx * stride_hw
        vals = tl.load(ptrs, mask=hw_mask, other=0.0)
        mean_val += tl.sum(vals)
        var_val += tl.sum(vals * vals)
    
    mean_val = mean_val / count
    var_val = var_val / count - mean_val * mean_val
    rstd = 1.0 / tl.sqrt(var_val + eps)
    
    # Apply normalization, tanh, hardswish, residual
    for c in range(C_PER_GROUP):
        c_idx = group_start + c
        gamma = tl.load(gamma_ptr + c_idx)
        beta = tl.load(beta_ptr + c_idx)
        
        ptrs = x_conv_ptr + pid_n * stride_n + c_idx * stride_c + hw_idx * stride_hw
        x_conv = tl.load(ptrs, mask=hw_mask, other=0.0)
        
        # Group normalization
        x_norm = (x_conv - mean_val) * rstd
        x_norm = x_norm * gamma + beta
        
        # Tanh using fast approximation
        exp_2x = tl.exp(2.0 * x_norm)
        x_tanh = (exp_2x - 1.0) / (exp_2x + 1.0)
        
        # HardSwish: x * relu6(x + 3) / 6
        x_plus_3 = x_tanh + 3.0
        relu6_val = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
        x_hard_swish = x_tanh * relu6_val * 0.16666667
        
        # Residual addition
        x_res = x_conv + x_hard_swish
        
        out_ptrs = output_ptr + pid_n * stride_n + c_idx * stride_c + hw_idx * stride_hw
        tl.store(out_ptrs, x_res, mask=hw_mask)


@triton.jit
def logsumexp_kernel(
    input_ptr, output_ptr,
    N, C, HW,
    stride_n, stride_c, stride_hw,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = pid // HW
    pid_hw = pid % HW
    
    # Find max for numerical stability
    max_val = -1e20
    for c_offset in range(0, C, BLOCK_C):
        c_idx = c_offset + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        ptrs = input_ptr + pid_n * stride_n + c_idx * stride_c + pid_hw * stride_hw
        vals = tl.load(ptrs, mask=c_mask, other=-1e20)
        block_max = tl.max(vals)
        max_val = tl.maximum(max_val, block_max)
    
    # Compute sum of exp(x - max)
    sum_exp = 0.0
    for c_offset in range(0, C, BLOCK_C):
        c_idx = c_offset + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        ptrs = input_ptr + pid_n * stride_n + c_idx * stride_c + pid_hw * stride_hw
        vals = tl.load(ptrs, mask=c_mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(c_mask, exp_vals, 0.0))
    
    result = max_val + tl.log(sum_exp)
    out_ptr = output_ptr + pid_n * HW + pid_hw
    tl.store(out_ptr, result)


def fused_groupnorm_tanh_hardswish_residual(x_conv, gamma, beta, groups, eps):
    N, C, H, W = x_conv.shape
    HW = H * W
    output = torch.empty_like(x_conv)
    
    C_per_group = C // groups
    BLOCK_HW = 256
    
    grid = (N, groups, triton.cdiv(HW, BLOCK_HW))
    fused_groupnorm_tanh_hardswish_residual_kernel[grid](
        x_conv, output,
        gamma, beta,
        N, C, HW, groups,
        eps,
        x_conv.stride(0), x_conv.stride(1), 1,
        BLOCK_HW=BLOCK_HW,
        C_PER_GROUP=C_per_group,
    )
    return output


def logsumexp_triton(x):
    N, C, H, W = x.shape
    HW = H * W
    output = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 128
    grid = (N * HW,)
    logsumexp_kernel[grid](
        x, output,
        N, C, HW,
        x.stride(0), x.stride(1), 1,
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
        x_conv = self.conv(x)
        x_res = fused_groupnorm_tanh_hardswish_residual(
            x_conv, self.gamma, self.beta, self.groups, self.eps
        )
        x_logsumexp = logsumexp_triton(x_res)
        return x_logsumexp
