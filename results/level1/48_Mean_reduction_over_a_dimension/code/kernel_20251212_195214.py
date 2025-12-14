import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_SIZE': 4, 'num_stages': 1, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 4, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 4, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'VEC_SIZE': 4, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['dim_size', 'inner_dim'],
)
@triton.jit
def mean_kernel_2d_optimized(
    x_ptr,
    output_ptr,
    outer_dim,
    inner_dim,
    dim_size,
    stride_outer,
    stride_inner,
    output_stride_outer,
    output_stride_inner,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """Optimized 2D mean reduction with vectorized loads and better memory patterns."""
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    
    if pid_outer >= outer_dim or pid_inner >= inner_dim:
        return
    
    # Output position
    output_offset = pid_outer * output_stride_outer + pid_inner * output_stride_inner
    
    # Vectorized accumulation
    sum_val = tl.zeros((VEC_SIZE,), dtype=tl.float32)
    block_start = 0
    
    # Process reduction dimension with vectorized loads
    while block_start < dim_size:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Compute input pointer with vectorization
        x_offset = pid_outer * stride_outer + pid_inner * stride_inner
        ptr = x_ptr + x_offset + offsets * stride_inner * inner_dim
        
        # Vectorized load
        if VEC_SIZE > 1:
            # Load multiple elements per thread
            vec_offsets = tl.arange(0, VEC_SIZE)[:, None] * stride_inner * inner_dim
            vec_ptrs = ptr + vec_offsets
            vec_mask = mask[None, :] & (vec_offsets < (dim_size - block_start) * stride_inner * inner_dim)[:, None]
            
            # Load vector elements
            vec_block = tl.load(vec_ptrs, mask=vec_mask, other=0.0)
            
            # Sum across vector dimension first
            vec_sum = tl.sum(vec_block, axis=0)
            block_sum = tl.sum(vec_sum)
        else:
            # Scalar load
            block = tl.load(ptr, mask=mask, other=0.0)
            block_sum = tl.sum(block)
        
        sum_val += block_sum
        block_start += BLOCK_SIZE * VEC_SIZE
    
    # Compute mean
    mean_val = tl.sum(sum_val) / dim_size
    tl.store(output_ptr + output_offset, mean_val)

@triton.autotune(
    configs=[
        triton.Config({'INNER_BLOCK': 512, 'REDUCE_BLOCK': 128, 'VEC_SIZE': 2, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'INNER_BLOCK': 256, 'REDUCE_BLOCK': 256, 'VEC_SIZE': 2, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'INNER_BLOCK': 128, 'REDUCE_BLOCK': 512, 'VEC_SIZE': 1, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'INNER_BLOCK': 64, 'REDUCE_BLOCK': 1024, 'VEC_SIZE': 1, 'num_stages': 1, 'num_warps': 4}),
    ],
    key=['inner_dim', 'dim_size', 'outer_dim'],
)
@triton.jit
def mean_kernel_3d_fast(
    x_ptr,
    output_ptr,
    outer_dim,
    inner_dim,
    dim_size,
    stride_outer,
    stride_inner,
    stride_reduce,
    output_stride_outer,
    output_stride_inner,
    INNER_BLOCK: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """Optimized 3D mean reduction with better parallelism and memory access patterns."""
    pid_outer = tl.program_id(0)
    pid_inner_block = tl.program_id(1)
    
    if pid_outer >= outer_dim:
        return
    
    # Inner dimension block with vectorization
    inner_start = pid_inner_block * INNER_BLOCK * VEC_SIZE
    inner_offsets = inner_start + tl.arange(0, INNER_BLOCK)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    inner_offsets_flat = tl.reshape(inner_offsets, (INNER_BLOCK * VEC_SIZE,))
    inner_mask = inner_offsets_flat < inner_dim
    
    # Accumulation registers for vectorized computation
    sum_vals = tl.zeros((INNER_BLOCK, VEC_SIZE), dtype=tl.float32)
    
    # Process reduction dimension in blocks for better cache utilization
    reduce_start = 0
    while reduce_start < dim_size:
        # Process multiple reduction steps per iteration with vectorization
        reduce_end = tl.minimum(reduce_start + REDUCE_BLOCK, dim_size)
        
        # Precompute base pointers to reduce address computation
        base_offset = pid_outer * stride_outer + reduce_start * stride_reduce
        
        # Vectorized loading of reduction dimension
        for k in range(0, REDUCE_BLOCK, VEC_SIZE):
            k_offset = reduce_start + k
            if k_offset >= reduce_end:
                break
            
            # Compute pointer with vectorized k dimension
            k_vec = tl.arange(0, VEC_SIZE)
            k_mask = (k_offset + k_vec) < reduce_end
            
            if tl.sum(k_mask) > 0:
                # Vectorized load across reduction dimension
                ptr_offsets = base_offset + k_vec[None, :] * stride_reduce + inner_offsets * stride_inner
                ptrs = x_ptr + tl.reshape(ptr_offsets, (INNER_BLOCK * VEC_SIZE * VEC_SIZE,))
                
                # Load with proper masking
                load_mask = inner_mask[:, None] & k_mask[None, :]
                load_mask_flat = tl.reshape(load_mask, (INNER_BLOCK * VEC_SIZE * VEC_SIZE,))
                
                loaded = tl.load(ptrs, mask=load_mask_flat, other=0.0)
                loaded_reshaped = tl.reshape(loaded, (INNER_BLOCK, VEC_SIZE, VEC_SIZE))
                
                # Accumulate
                sum_vals += tl.sum(loaded_reshaped, axis=2)
        
        reduce_start += REDUCE_BLOCK
    
    # Sum across vector dimension
    sum_vals_flat = tl.sum(sum_vals, axis=1)
    
    # Compute mean
    mean_vals = sum_vals_flat / dim_size
    
    # Store results with proper masking
    output_offsets = pid_outer * output_stride_outer + (inner_start + tl.arange(0, INNER_BLOCK * VEC_SIZE)) * output_stride_inner
    tl.store(output_ptr + output_offsets, mean_vals, mask=inner_mask)

def triton_mean_optimized(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Optimized Triton implementation of mean reduction with autotuning."""
    if x.dim() == 0:
        return x.clone()
    
    # Handle negative dimensions
    if dim < 0:
        dim = x.dim() + dim
    
    input_shape = x.shape
    dim_size = input_shape[dim]
    
    # Handle trivial cases
    if dim_size == 1:
        return x.squeeze(dim)
    
    # Compute output shape
    output_shape = list(input_shape)
    output_shape.pop(dim)
    output_shape = tuple(output_shape)
    
    # Create output tensor
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # For 1D tensors, use simple reduction
    if x.dim() == 1:
        return torch.mean(x, dim=dim)
    
    # For 2D tensors
    if x.dim() == 2:
        if dim == 0:
            outer_dim = 1
            inner_dim = input_shape[1]
            stride_outer = input_shape[1] * dim_size
            stride_inner = 1
        else:  # dim == 1
            outer_dim = input_shape[0]
            inner_dim = 1
            stride_outer = dim_size
            stride_inner = 1
        
        output_stride_outer = inner_dim
        output_stride_inner = 1
        
        grid = (outer_dim, inner_dim)
        mean_kernel_2d_optimized[grid](
            x, output,
            outer_dim, inner_dim, dim_size,
            stride_outer, stride_inner,
            output_stride_outer, output_stride_inner,
        )
        
        return output
    
    # For 3D tensors
    if x.dim() == 3:
        if dim == 0:
            outer_dim = 1
            inner_dim = input_shape[1] * input_shape[2]
            stride_outer = input_shape[1] * input_shape[2] * dim_size
            stride_inner = 1
            stride_reduce = input_shape[1] * input_shape[2]
        elif dim == 1:
            outer_dim = input_shape[0]
            inner_dim = input_shape[2]
            stride_outer = input_shape[1] * input_shape[2]
            stride_inner = 1
            stride_reduce = input_shape[2]
        else:  # dim == 2
            outer_dim = input_shape[0] * input_shape[1]
            inner_dim = 1
            stride_outer = input_shape[2] * dim_size
            stride_inner = 1
            stride_reduce = 1
        
        output_stride_outer = inner_dim
        output_stride_inner = 1
        
        # Optimize grid for SM count (128 SMs on 4090)
        # Target 2-4 blocks per SM for better utilization
        num_inner_blocks = triton.cdiv(inner_dim, 512)  # Start with larger blocks
        grid_size = outer_dim * num_inner_blocks
        
        if grid_size < 256:  # Too few blocks
            # Adjust block size to increase parallelism
            num_inner_blocks = triton.cdiv(inner_dim, 128)
        
        grid = (outer_dim, num_inner_blocks)
        mean_kernel_3d_fast[grid](
            x, output,
            outer_dim, inner_dim, dim_size,
            stride_outer, stride_inner, stride_reduce,
            output_stride_outer, output_stride_inner,
        )
        
        return output.reshape(output_shape)
    
    # For higher dimensions, fall back to PyTorch
    return torch.mean(x, dim=dim)

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension using Triton.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.
        Uses optimized Triton kernels for 2D and 3D tensors.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        return triton_mean_optimized(x, self.dim)
