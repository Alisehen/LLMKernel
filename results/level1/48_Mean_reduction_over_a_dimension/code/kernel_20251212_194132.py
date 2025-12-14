import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel_2d(
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
):
    """Kernel for 2D mean reduction with optimized memory access patterns."""
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    
    if pid_outer >= outer_dim or pid_inner >= inner_dim:
        return
    
    # Base pointer for this output position
    output_offset = pid_outer * output_stride_outer + pid_inner * output_stride_inner
    
    # Accumulate sum using block-wise reduction
    sum_val = 0.0
    block_start = 0
    
    # Process reduction dimension in blocks
    while block_start < dim_size:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < dim_size
        
        # Compute input pointer for this block
        x_offset = pid_outer * stride_outer + pid_inner * stride_inner
        ptr = x_ptr + x_offset + offsets * stride_inner * inner_dim
        
        # Load block
        block = tl.load(ptr, mask=mask, other=0.0)
        
        # Accumulate sum
        block_sum = tl.sum(block, axis=0)
        sum_val += block_sum
        
        block_start += BLOCK_SIZE
    
    # Compute mean
    mean_val = sum_val / dim_size
    tl.store(output_ptr + output_offset, mean_val)

@triton.jit
def mean_kernel_3d(
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
    INNER_BLOCK_SIZE: tl.constexpr,
):
    """Kernel for 3D mean reduction with optimized memory access and vectorization."""
    pid_outer = tl.program_id(0)
    pid_inner_block = tl.program_id(1)
    
    if pid_outer >= outer_dim:
        return
    
    # Inner dimension block processing
    inner_start = pid_inner_block * INNER_BLOCK_SIZE
    inner_offsets = inner_start + tl.arange(0, INNER_BLOCK_SIZE)
    inner_mask = inner_offsets < inner_dim
    
    # Initialize accumulation register
    sum_vals = tl.zeros((INNER_BLOCK_SIZE,), dtype=tl.float32)
    
    # Reduce over the specified dimension
    for k in range(dim_size):
        # Compute base pointer for this reduction step
        base_offset = pid_outer * stride_outer + k * stride_reduce
        
        # Load block with vectorization
        ptrs = x_ptr + base_offset + inner_offsets * stride_inner
        block = tl.load(ptrs, mask=inner_mask, other=0.0)
        
        # Accumulate
        sum_vals += block
    
    # Compute mean
    mean_vals = sum_vals / dim_size
    
    # Store results with mask
    output_offsets = pid_outer * output_stride_outer + (inner_start + tl.arange(0, INNER_BLOCK_SIZE)) * output_stride_inner
    tl.store(output_ptr + output_offsets, mean_vals, mask=inner_mask)

def triton_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Optimized Triton implementation of mean reduction."""
    if x.dim() == 0:
        return x.clone()
    
    # Handle negative dimensions
    if dim < 0:
        dim = x.dim() + dim
    
    # Get input shape and strides
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
        # Handle 1D case with custom kernel or fallback to PyTorch
        # For simplicity, use PyTorch for 1D
        return torch.mean(x, dim=dim)
    
    # For 2D tensors (batch_size × features)
    if x.dim() == 2:
        if dim == 0:
            # Reduce over batch dimension
            outer_dim = 1
            inner_dim = input_shape[1]
            stride_outer = input_shape[1] * dim_size
            stride_inner = 1
        else:  # dim == 1
            # Reduce over feature dimension
            outer_dim = input_shape[0]
            inner_dim = 1
            stride_outer = dim_size
            stride_inner = 1
        
        output_stride_outer = inner_dim
        output_stride_inner = 1
        
        # Choose block size based on reduction dimension
        BLOCK_SIZE = 1024 if dim_size >= 1024 else triton.next_power_of_2(dim_size)
        if BLOCK_SIZE < 16:
            BLOCK_SIZE = 16
        
        grid = (outer_dim, inner_dim)
        mean_kernel_2d[grid](
            x, output,
            outer_dim, inner_dim, dim_size,
            stride_outer, stride_inner,
            output_stride_outer, output_stride_inner,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
    
    # For 3D tensors (batch_size × dim1 × dim2)
    if x.dim() == 3:
        # Reshape to handle arbitrary reduction dimension
        if dim == 0:
            # Reduce over batch
            outer_dim = 1
            inner_dim = input_shape[1] * input_shape[2]
            stride_outer = input_shape[1] * input_shape[2] * dim_size
            stride_inner = 1
            stride_reduce = input_shape[1] * input_shape[2]
        elif dim == 1:
            # Reduce over first feature dimension
            outer_dim = input_shape[0]
            inner_dim = input_shape[2]
            stride_outer = input_shape[1] * input_shape[2]
            stride_inner = 1
            stride_reduce = input_shape[2]
        else:  # dim == 2
            # Reduce over second feature dimension
            outer_dim = input_shape[0] * input_shape[1]
            inner_dim = 1
            stride_outer = input_shape[2] * dim_size
            stride_inner = 1
            stride_reduce = 1
        
        output_stride_outer = inner_dim
        output_stride_inner = 1
        
        # Optimize block sizes for A100
        INNER_BLOCK_SIZE = 256  # Process multiple inner dimensions together
        
        grid = (outer_dim, triton.cdiv(inner_dim, INNER_BLOCK_SIZE))
        mean_kernel_3d[grid](
            x, output,
            outer_dim, inner_dim, dim_size,
            stride_outer, stride_inner, stride_reduce,
            output_stride_outer, output_stride_inner,
            INNER_BLOCK_SIZE=INNER_BLOCK_SIZE
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
        return triton_mean(x, self.dim)
