import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'VEC_SIZE': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_SIZE': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 8}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_SIZE': 4}, num_warps=16, num_stages=4),
    ],
    key=['total_elements']
)
@triton.jit
def clamp_div_kernel(
    x_ptr,
    out_ptr,
    N, C, D, H, W,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    min_value,
    divisor,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Optimized fused clamp(min=min_value) + division by divisor
    Uses vectorized loads/stores for better memory throughput
    """
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block with vectorization
    block_start = pid * BLOCK_SIZE * VEC_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    offsets = tl.reshape(offsets, -1)
    
    # Create mask for vectorized loads/stores
    mask = offsets < total_elements
    
    # Fast path for contiguous tensors (common case after ConvTranspose3d)
    # Use precomputed strides for 5D index calculation only when needed
    # Optimized memory offset calculation
    HW = H * W
    DHW = D * HW
    CDHW = C * DHW
    
    # Compute 5D indices from 1D offsets using efficient integer arithmetic
    idx = offsets
    n_idx = tl.math.floor(idx / CDHW).to(tl.int64)
    remainder = idx - n_idx * CDHW
    c_idx = tl.math.floor(remainder / DHW).to(tl.int64)
    remainder = remainder - c_idx * DHW
    d_idx = tl.math.floor(remainder / HW).to(tl.int64)
    remainder = remainder - d_idx * HW
    h_idx = tl.math.floor(remainder / W).to(tl.int64)
    w_idx = remainder - h_idx * W
    
    # Compute memory offsets using strides - optimized to reduce register pressure
    ptr_offsets = (n_idx * stride_xn + 
                   c_idx * stride_xc + 
                   d_idx * stride_xd + 
                   h_idx * stride_xh + 
                   w_idx * stride_xw)
    
    # Vectorized load
    x = tl.load(x_ptr + ptr_offsets, mask=mask, other=0.0)
    
    # Fused operations: clamp + division
    # Use maximum for clamp(min=min_value) - single operation
    # Division by divisor can be multiplication by reciprocal if divisor is constant
    x = tl.maximum(x, min_value)
    
    # Use multiplication by reciprocal if divisor is not zero
    # This is faster than division on most GPUs
    inv_divisor = 1.0 / divisor
    x = x * inv_divisor
    
    # Vectorized store
    tl.store(out_ptr + ptr_offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=3),
    ],
    key=['total_elements']
)
@triton.jit
def clamp_div_kernel_contiguous(
    x_ptr,
    out_ptr,
    min_value,
    divisor,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for contiguous tensors (common case)
    Uses simplified indexing and vectorization
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load directly using 1D offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations
    x = tl.maximum(x, min_value)
    inv_divisor = 1.0 / divisor
    x = x * inv_divisor
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_clamp_div(x, min_value, divisor):
    """
    Fused clamp(min=min_value) + division by divisor
    Chooses optimized kernel based on tensor contiguity
    """
    N, C, D, H, W = x.shape
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Calculate total elements
    total_elements = N * C * D * H * W
    
    # Check if tensor is contiguous in memory
    # This is common after ConvTranspose3d operations
    is_contiguous = x.is_contiguous()
    
    if is_contiguous:
        # Use optimized contiguous kernel
        grid = lambda META: (triton.cdiv(total_elements, META['BLOCK_SIZE']),)
        clamp_div_kernel_contiguous[grid](
            x, out,
            float(min_value), float(divisor),
            total_elements,
        )
    else:
        # Use general kernel with 5D indexing
        grid = lambda META: (triton.cdiv(total_elements, META['BLOCK_SIZE'] * META['VEC_SIZE']),)
        clamp_div_kernel[grid](
            x, out,
            N, C, D, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
            float(min_value), float(divisor),
            total_elements,
        )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused clamp + division (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        # Step 2: Fused clamp + division in Triton
        x = fused_clamp_div(x, self.min_value, self.divisor)
        return x
