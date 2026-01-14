import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_bn_avgpool_kernel_v2(
    x_ptr, partial_sums_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    num_blocks_per_nc,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a portion of one (batch, channel) pair
    pid = tl.program_id(0)
    nc_idx = pid // num_blocks_per_nc
    block_idx = pid % num_blocks_per_nc
    
    batch_idx = nc_idx // C
    channel_idx = nc_idx % C
    
    # Load batch norm parameters for this channel
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Precompute batch norm coefficients with scale fused in
    inv_std = 1.0 / tl.sqrt(var + eps)
    coeff = scale_factor * gamma * inv_std
    bias = beta - mean * gamma * inv_std
    
    total_elements = D * H * W
    
    # Each block processes a range of elements
    start_idx = block_idx * BLOCK_SIZE
    offs = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Convert linear index to 3D indices
    d_idx = offs // (H * W)
    hw_rem = offs % (H * W)
    h_idx = hw_rem // W
    w_idx = hw_rem % W
    
    # Compute pointers
    base_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    ptrs = base_ptr + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
    
    # Load values
    x_vals = tl.load(ptrs, mask=mask, other=0.0)
    
    # Apply fused scale + batch norm
    y_vals = x_vals * coeff + bias
    
    # Compute partial sum
    partial_sum = tl.sum(tl.where(mask, y_vals, 0.0), axis=0)
    
    # Store partial sum
    partial_idx = nc_idx * num_blocks_per_nc + block_idx
    tl.store(partial_sums_ptr + partial_idx, partial_sum)


@triton.jit
def reduce_partial_sums_kernel(
    partial_sums_ptr, out_ptr,
    N, C, num_blocks_per_nc, total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    nc_idx = pid
    
    # Load and sum all partial sums for this (batch, channel)
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    base_ptr = partial_sums_ptr + nc_idx * num_blocks_per_nc
    
    for start in range(0, num_blocks_per_nc, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_blocks_per_nc
        vals = tl.load(base_ptr + offs, mask=mask, other=0.0)
        accumulator += vals
    
    total_sum = tl.sum(accumulator, axis=0)
    avg = total_sum / total_elements
    
    # Store result
    out_offset = batch_idx * C + channel_idx
    tl.store(out_ptr + out_offset, avg)


@triton.jit
def fused_scale_bn_avgpool_single_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // C
    channel_idx = pid % C
    
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    coeff = scale_factor * gamma * inv_std
    bias = beta - mean * gamma * inv_std
    
    total_elements = D * H * W
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    base_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    
    for start in range(0, total_elements, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elements
        
        d_idx = offs // (H * W)
        hw_rem = offs % (H * W)
        h_idx = hw_rem // W
        w_idx = hw_rem % W
        
        ptrs = base_ptr + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x_vals = tl.load(ptrs, mask=mask, other=0.0)
        
        y_vals = x_vals * coeff + bias
        accumulator += tl.where(mask, y_vals, 0.0)
    
    total_sum = tl.sum(accumulator, axis=0)
    avg = total_sum / total_elements
    
    out_offset = batch_idx * C + channel_idx
    tl.store(out_ptr + out_offset, avg)


def fused_scale_bn_avgpool(x, gamma, beta, running_mean, running_var, scale_factor, eps):
    N, C, D, H, W = x.shape
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    total_elements = D * H * W
    BLOCK_SIZE = 1024
    
    # Use multi-block approach for large spatial dimensions
    if total_elements > 4096:
        num_blocks_per_nc = triton.cdiv(total_elements, BLOCK_SIZE)
        partial_sums = torch.empty((N * C * num_blocks_per_nc,), device=x.device, dtype=torch.float32)
        
        grid1 = (N * C * num_blocks_per_nc,)
        fused_scale_bn_avgpool_kernel_v2[grid1](
            x, partial_sums,
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            num_blocks_per_nc,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        REDUCE_BLOCK = min(256, triton.next_power_of_2(num_blocks_per_nc))
        grid2 = (N * C,)
        reduce_partial_sums_kernel[grid2](
            partial_sums, out.view(-1),
            N, C, num_blocks_per_nc, total_elements,
            BLOCK_SIZE=REDUCE_BLOCK,
        )
    else:
        grid = (N * C,)
        fused_scale_bn_avgpool_single_kernel[grid](
            x, out.view(-1),
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            BLOCK_SIZE=BLOCK_SIZE,
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
