import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_bn_avgpool_atomic_kernel(
    x_ptr, out_ptr,
    gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    scale_factor, eps,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    # 2D grid: (N*C, NUM_BLOCKS)
    nc_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
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
    
    # Each block processes a range of elements
    start_idx = block_idx * BLOCK_SIZE
    offs = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Convert linear index to 3D indices
    HW = H * W
    d_idx = offs // HW
    hw_rem = offs % HW
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
    
    # Atomic add to output (will be divided later)
    tl.atomic_add(out_ptr + nc_idx, partial_sum)


@triton.jit
def divide_kernel(
    out_ptr,
    divisor,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS
    
    vals = tl.load(out_ptr + offs, mask=mask)
    vals = vals / divisor
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def fused_scale_bn_avgpool_single_kernel_opt(
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
    
    gamma = tl.load(gamma_ptr + channel_idx)
    beta = tl.load(beta_ptr + channel_idx)
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    coeff = scale_factor * gamma * inv_std
    bias = beta - mean * gamma * inv_std
    
    HW = H * W
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    base_ptr = x_ptr + batch_idx * stride_n + channel_idx * stride_c
    
    num_iters = tl.cdiv(total_elements, BLOCK_SIZE)
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
        accumulator += tl.where(mask, y_vals, 0.0)
    
    total_sum = tl.sum(accumulator, axis=0)
    avg = total_sum / total_elements
    
    tl.store(out_ptr + pid, avg)


def fused_scale_bn_avgpool(x, gamma, beta, running_mean, running_var, scale_factor, eps):
    N, C, D, H, W = x.shape
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    total_elements = D * H * W
    
    # For RTX 4090, use larger blocks
    if total_elements <= 8192:
        # Single kernel approach for smaller inputs
        BLOCK_SIZE = 1024
        grid = (N * C,)
        fused_scale_bn_avgpool_single_kernel_opt[grid](
            x, out.view(-1),
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Multi-block with atomic reduction
        BLOCK_SIZE = 2048
        NUM_BLOCKS = triton.cdiv(total_elements, BLOCK_SIZE)
        
        # Initialize output to zero for atomic adds
        out_flat = out.view(-1)
        out_flat.zero_()
        
        grid = (N * C, NUM_BLOCKS)
        fused_scale_bn_avgpool_atomic_kernel[grid](
            x, out_flat,
            gamma, beta, running_mean, running_var,
            scale_factor, eps,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            total_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BLOCKS=NUM_BLOCKS,
        )
        
        # Divide by total elements
        n_elements = N * C
        DIV_BLOCK = 256
        div_grid = (triton.cdiv(n_elements, DIV_BLOCK),)
        divide_kernel[div_grid](
            out_flat,
            float(total_elements),
            N_ELEMENTS=n_elements,
            BLOCK_SIZE=DIV_BLOCK,
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
