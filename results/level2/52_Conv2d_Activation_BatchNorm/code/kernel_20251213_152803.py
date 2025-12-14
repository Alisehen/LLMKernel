import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    output_ptr,
    M,  # Batch * Channels
    N,  # Height * Width
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized 2D kernel with better grid layout for 4D tensors"""
    # 2D grid: (batch*channels, height*width)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for 2D block
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for boundaries
    mask_m = rm < M
    mask_n = rn < N
    
    # Load input with 2D indexing
    x_ptrs = x_ptr + rm[:, None] * stride_m + rn[None, :] * stride_n
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute fused activation
    # Replaced tl.log1p with manual implementation for Triton compatibility
    # softplus(x) = log(1 + exp(x))
    # Use numerical stability optimization for both cases
    is_positive = x > 0
    # For x > 0: softplus(x) = x + log(1 + exp(-x))
    # For x <= 0: softplus(x) = log(1 + exp(x))
    
    # Handle positive case
    pos_term = tl.where(is_positive, x, 0.0)
    neg_x = tl.where(is_positive, -x, 0.0)
    exp_neg_x = tl.exp(neg_x)
    pos_softplus = pos_term + tl.log(1.0 + exp_neg_x)
    
    # Handle negative case
    neg_term = tl.where(~is_positive, x, 0.0)
    exp_x = tl.exp(neg_term)
    neg_softplus = tl.log(1.0 + exp_x)
    
    # Combine both cases
    softplus_val = tl.where(is_positive, pos_softplus, neg_softplus)
    
    # tanh of softplus
    tanh_val = tl.tanh(softplus_val)
    
    # Multiply with original x
    output = x * tanh_val
    
    # Store result to output tensor
    output_ptrs = output_ptr + rm[:, None] * stride_m + rn[None, :] * stride_n
    tl.store(output_ptrs, output, mask=mask)

def triton_fused_activation(x: torch.Tensor) -> torch.Tensor:
    """Wrapper with 2D grid for 4D tensors"""
    # Reshape 4D tensor to (batch*channels, height*width)
    B, C, H, W = x.shape
    M = B * C
    N = H * W
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Use optimized 2D grid layout
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel with optimized block sizes
    fused_activation_kernel[grid](
        x, output, M, N,
        x.stride(0) * C,  # stride for batch*channels dimension
        x.stride(2),      # stride for spatial dimensions
        BLOCK_M=32,       # Optimized for spatial locality
        BLOCK_N=64        # Good balance for Ada Lovelace
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused activation using optimized Triton kernel
        x = triton_fused_activation(x)
        
        # Apply batch normalization
        x = self.bn(x)
        return x
