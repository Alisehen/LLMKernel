import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def im2col_kernel(
    x_ptr, col_ptr,
    batch, in_channels, in_h, in_w,
    out_h, out_w, kernel_size,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_cb, stride_cc, stride_cs,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of spatial positions
    pid = tl.program_id(0)
    
    # Total spatial positions across batch
    total_spatial = batch * out_h * out_w
    
    # Compute batch, oh, ow from pid
    spatial_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    b = spatial_idx // (out_h * out_w)
    rem = spatial_idx % (out_h * out_w)
    oh = rem // out_w
    ow = rem % out_w
    
    mask = spatial_idx < total_spatial
    
    # For each spatial position, extract all kernel_size^2 * in_channels values
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = oh + kh
                iw = ow + kw
                
                # Load input value
                x_idx = b * stride_xb + ic * stride_xc + ih * stride_xh + iw * stride_xw
                x_val = tl.load(x_ptr + x_idx, mask=mask & (ih < in_h) & (iw < in_w), other=0.0)
                
                # Store to column matrix
                col_idx_c = ic * kernel_size * kernel_size + kh * kernel_size + kw
                col_idx = spatial_idx * stride_cs + col_idx_c * stride_cc
                tl.store(col_ptr + col_idx, x_val, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = acc.to(tl.float32)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def add_bias_kernel(
    x_ptr, bias_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    result = x + bias[None, :]
    
    out_ptrs = output_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    tl.store(out_ptrs, result, mask=mask)


@triton.jit
def fused_avgpool_sigmoid_sum_kernel(
    x_ptr, output_ptr,
    batch, out_channels, conv_h, conv_w,
    pool_h, pool_w, pool_kernel,
    stride_xb, stride_xc, stride_xh, stride_xw,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    b = pid_b
    c_start = pid_c * BLOCK_C
    c_offs = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offs < out_channels
    
    # Accumulator for sum across spatial dimensions
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Iterate over pooled spatial locations
    for ph in range(pool_h):
        for pw in range(pool_w):
            pool_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
            pool_count = 0.0
            
            # Average pooling window
            for kh in range(pool_kernel):
                for kw in range(pool_kernel):
                    conv_h_idx = ph * pool_kernel + kh
                    conv_w_idx = pw * pool_kernel + kw
                    
                    if conv_h_idx < conv_h and conv_w_idx < conv_w:
                        x_idx = b * stride_xb + c_offs * stride_xc + conv_h_idx * stride_xh + conv_w_idx * stride_xw
                        x_val = tl.load(x_ptr + x_idx, mask=c_mask, other=0.0)
                        pool_sum += x_val
                        pool_count += 1.0
            
            # Average and sigmoid
            if pool_count > 0.0:
                avg_val = pool_sum / pool_count
                sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
                acc += sigmoid_val
    
    # Store result
    out_idx = b * out_channels + c_offs
    tl.store(output_ptr + out_idx, acc, mask=c_mask)


def optimized_conv2d(x, weight, bias):
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    out_h = in_h - kernel_size + 1
    out_w = in_w - kernel_size + 1
    
    # Im2col transformation
    col_h = kernel_size * kernel_size * in_channels
    col_w = batch * out_h * out_w
    col = torch.empty((col_w, col_h), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 256
    grid_im2col = (triton.cdiv(col_w, BLOCK_SIZE),)
    im2col_kernel[grid_im2col](
        x, col,
        batch, in_channels, in_h, in_w,
        out_h, out_w, kernel_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        0, 1, col_h,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape weight for matrix multiplication
    weight_reshaped = weight.reshape(out_channels, -1).t()  # (col_h, out_channels)
    
    # Matrix multiplication: col @ weight_reshaped
    output = torch.empty((col_w, out_channels), device=x.device, dtype=x.dtype)
    
    M, K = col_w, col_h
    K2, N = weight_reshaped.shape
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid_matmul](
        col, weight_reshaped, output,
        M, N, K,
        col.stride(0), col.stride(1),
        weight_reshaped.stride(0), weight_reshaped.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Add bias
    grid_bias = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    add_bias_kernel[grid_bias](
        output, bias, output,
        M, N,
        output.stride(0), output.stride(1),
        BLOCK_M=64, BLOCK_N=64,
    )
    
    # Reshape to (batch, out_channels, out_h, out_w)
    output = output.reshape(batch, out_h, out_w, out_channels).permute(0, 3, 1, 2).contiguous()
    
    return output


def fused_forward(x, weight, bias, pool_kernel_size):
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    
    # Convolution
    conv_out = optimized_conv2d(x, weight, bias)
    
    _, _, conv_h, conv_w = conv_out.shape
    pool_h = conv_h // pool_kernel_size
    pool_w = conv_w // pool_kernel_size
    
    # Fused avgpool + sigmoid + sum
    output = torch.empty((batch, out_channels), device=x.device, dtype=x.dtype)
    
    BLOCK_C = 64
    grid_pool = (batch, triton.cdiv(out_channels, BLOCK_C))
    fused_avgpool_sigmoid_sum_kernel[grid_pool](
        conv_out, output,
        batch, out_channels, conv_h, conv_w,
        pool_h, pool_w, pool_kernel_size,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        BLOCK_C=BLOCK_C,
    )
    
    # Sum over channels
    result = output.sum(dim=1)
    return result


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        return fused_forward(x, self.weight, self.bias, self.pool_kernel_size)
