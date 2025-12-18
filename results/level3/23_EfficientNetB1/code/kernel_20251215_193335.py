import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_pointwise_conv_bn_relu6_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    N, C_in, C_out, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc,
    stride_on, stride_oc, stride_oh, stride_ow,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """Fused pointwise convolution + batch norm + ReLU6 - OPTIMIZED"""
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Create base offsets with minimal computation
    n_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    oc_offs = pid_oc * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    hw_offs = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    
    # Masks - compute once
    n_mask = n_offs < N
    oc_mask = oc_offs < C_out
    hw_mask = hw_offs < H * W
    
    # Compute spatial indices (fused to reduce register usage)
    h_idx = tl.where(hw_mask, hw_offs // W, 0)
    w_idx = tl.where(hw_mask, hw_offs % W, 0)
    hw_mask = hw_mask & (h_idx < H) & (w_idx < W)
    
    # Initialize accumulator - smaller footprint
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Loop strategy: process input channels in largest possible blocks
    for c_block in range(0, C_in, 128):  # Increased for better memory coalescing
        c_offs = c_block + tl.arange(0, 128)
        c_mask = c_offs < C_in
        
        # Load weight block - preload for reuse
        w_ptrs = w_ptr + (oc_offs[:, None] * stride_wn + c_offs[None, :] * stride_wc)
        w_block = tl.load(w_ptrs, mask=oc_mask[:, None] & c_mask[None, :], other=0.0)
        
        # Process spatial chunks within this weight block
        for hw_chunk in range(0, BLOCK_SIZE_HW, 32):  # Smaller chunks for register efficiency
            hw_chunk_offs = hw_offs + hw_chunk
            hw_chunk_mask = hw_mask & (hw_chunk_offs < H * W)
            
            if tl.sum(hw_chunk_mask) == 0:
                continue
            
            # Compute spatial indices for this chunk
            h_chunk_idx = tl.where(hw_chunk_mask, hw_chunk_offs // W, 0)
            w_chunk_idx = tl.where(hw_chunk_mask, hw_chunk_offs % W, 0)
            
            # Load input block for this chunk
            x_ptrs = x_ptr + (
                n_offs[:, None, None] * stride_xn +
                c_offs[None, :, None] * stride_xc +
                h_chunk_idx[None, None, :] * stride_xh +
                w_chunk_idx[None, None, :] * stride_xw
            )
            x_block = tl.load(
                x_ptrs,
                mask=n_mask[:, None, None] & c_mask[None, :, None] & hw_chunk_mask[None, None, :],
                other=0.0
            )
            
            # Accumulate - using efficient dot product
            chunk_acc = tl.sum(x_block * w_block[None, :, :, None], axis=2)
            acc += tl.where(hw_chunk_mask[None, None, 0], chunk_acc, 0.0)
    
    # Load batch norm parameters once
    bn_weight = tl.load(bn_weight_ptr + oc_offs, mask=oc_mask, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    bn_mean = tl.load(bn_mean_ptr + oc_offs, mask=oc_mask, other=0.0)
    bn_var = tl.load(bn_var_ptr + oc_offs, mask=oc_mask, other=1.0)
    
    # Fused batch norm: avoid storing intermediate
    inv_std = bn_weight * tl.rsqrt(bn_var + eps)
    normalized = (acc - bn_mean[None, :]) * inv_std[None, :] + bn_bias[None, :]
    
    # ReLU6: min(max(x, 0), 6) - compute directly
    activated = tl.minimum(tl.maximum(normalized, 0.0), 6.0)
    
    # Store output in spatial chunks to reduce register pressure
    for hw_chunk in range(0, BLOCK_SIZE_HW, 32):
        hw_chunk_offs = hw_offs + hw_chunk
        hw_chunk_mask = hw_mask & (hw_chunk_offs < H * W)
        
        if tl.sum(hw_chunk_mask) == 0:
            continue
        
        h_chunk_idx = tl.where(hw_chunk_mask, hw_chunk_offs // W, 0)
        w_chunk_idx = tl.where(hw_chunk_mask, hw_chunk_offs % W, 0)
        
        out_ptrs = out_ptr + (
            n_offs[:, None] * stride_on +
            oc_offs[None, :] * stride_oc +
            h_chunk_idx[None, :] * stride_oh +
            w_chunk_idx[None, :] * stride_ow
        )
        
        tl.store(
            out_ptrs,
            activated,
            mask=n_mask[:, None] & oc_mask[None, :] & hw_chunk_mask[None, :]
        )


def fused_pointwise_conv_bn_relu6(x, weight, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Fused pointwise convolution, batch norm, and ReLU6 with autotuning."""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Create output tensor
    out = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    
    # Grid configuration with autotuning
    configs = [
        {'BLOCK_SIZE_N': 1, 'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 64},   # Conservative
        {'BLOCK_SIZE_N': 2, 'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 64},   # Balanced
        {'BLOCK_SIZE_N': 4, 'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_HW': 128},  # Aggressive
    ]
    
    for config in configs:
        BLOCK_SIZE_N = min(config['BLOCK_SIZE_N'], N)
        BLOCK_SIZE_C = min(config['BLOCK_SIZE_C'], C_out)
        BLOCK_SIZE_HW = min(config['BLOCK_SIZE_HW'], H * W)
        
        grid = (
            triton.cdiv(N, BLOCK_SIZE_N),
            triton.cdiv(C_out, BLOCK_SIZE_C),
            triton.cdiv(H * W, BLOCK_SIZE_HW),
        )
        
        fused_pointwise_conv_bn_relu6_kernel[grid](
            x, weight, out,
            bn_weight, bn_bias, bn_mean, bn_var,
            N, C_in, C_out, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )
    
    return out


@triton.autotune(
    configs=[
        {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},  # Low register pressure
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},  # Balanced
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  # Aggressive
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_bias_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused linear layer with bias - OPTIMIZED"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Tile offsets with proper masking
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pre-compute bias pointer
    bias_ptrs = b_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    
    # Main accumulation loop with better memory access pattern
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Load input tile
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight tile
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Matrix multiplication with TF32 tensor cores
        acc += tl.dot(x, w, allow_tf32=True)
    
    # Add bias and store
    result = acc + bias[None, :]
    
    # Store with proper masking
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


def fused_linear_bias(x, weight, bias):
    """Fused linear layer with bias using autotuned kernel."""
    M, K = x.shape
    N = weight.shape[0]
    
    # Ensure inputs are contiguous and weight is transposed
    x = x.contiguous()
    weight = weight.contiguous().t()  # (K, N)
    bias = bias.contiguous()
    
    # Create output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel with autotuning
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_linear_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1),
    )
    
    return out


class FusedMBConvBlock(nn.Module):
    """Fused MBConv block with Triton optimizations."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(FusedMBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = round(in_channels * expand_ratio)
        
        # Expansion phase (fused: conv + bn + relu6)
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim,
                                       kernel_size=3, stride=stride,
                                       padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        # Projection phase
        self.project_conv = nn.Conv2d(hidden_dim, out_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        # Fused expansion phase using Triton
        if self.expand_ratio != 1:
            x = fused_pointwise_conv_bn_relu6(
                x,
                self.expand_conv.weight,
                self.expand_bn.weight,
                self.expand_bn.bias,
                self.expand_bn.running_mean,
                self.expand_bn.running_var,
                self.expand_bn.eps
            )
        
        # Depthwise convolution (keep as PyTorch for now)
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = F.relu6(x, inplace=True)
        
        # Projection phase
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        return x


class ModelNew(nn.Module):
    """EfficientNetB1 with Triton optimizations."""
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks with Triton fusion
        self.mbconv1 = FusedMBConvBlock(32, 16, 1, 1)
        self.mbconv2 = FusedMBConvBlock(16, 24, 2, 6)
        self.mbconv3 = FusedMBConvBlock(24, 40, 2, 6)
        self.mbconv4 = FusedMBConvBlock(40, 80, 2, 6)
        self.mbconv5 = FusedMBConvBlock(80, 112, 1, 6)
        self.mbconv6 = FusedMBConvBlock(112, 192, 2, 6)
        self.mbconv7 = FusedMBConvBlock(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer (will use Triton kernel)
        self.fc_weight = nn.Parameter(torch.randn(num_classes, 1280))
        self.fc_bias = nn.Parameter(torch.randn(num_classes))
        
    def forward(self, x):
        # Initial conv + bn + relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        # Final conv + bn + relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # Global pooling and flatten
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Final classification layer (fused linear + bias)
        x = fused_linear_bias(x, self.fc_weight, self.fc_bias)
        
        return x
