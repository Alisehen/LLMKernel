import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,
    norm_ptr,
    batch_size,
    dim,
    stride_x_batch,
    stride_x_dim,
    stride_norm_batch,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute L2 norm by summing squared values across dimension."""
    # 1D launch grid
    pid = tl.program_id(0)
    
    # Calculate batch and dimension indices from single pid
    num_batch_blocks = tl.cdiv(batch_size, BLOCK_B)
    batch_block_idx = pid // num_batch_blocks
    dim_block_idx = pid % num_batch_blocks
    
    # Batch offsets for this block
    batch_start = batch_block_idx * BLOCK_B
    batch_offsets = batch_start + tl.arange(0, BLOCK_B)
    batch_mask = batch_offsets < batch_size
    
    # Dimension offsets for this block
    dim_start = dim_block_idx * BLOCK_D
    dim_offsets = dim_start + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < dim
    
    # Initialize local sum
    local_sum = tl.zeros((BLOCK_B,), dtype=tl.float32)
    
    if tl.sum(dim_mask) > 0:
        # Create pointers for this block
        x_ptrs = x_ptr + (
            batch_offsets[:, None] * stride_x_batch +
            dim_offsets[None, :] * stride_x_dim
        )
        
        # Load with proper masking
        mask = batch_mask[:, None] & dim_mask[None, :]
        x_block = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Accumulate sum of squares
        local_sum = tl.sum(x_block * x_block, axis=1)
    
    # Atomic add to global norm tensor with proper masking
    if tl.sum(batch_mask) > 0:
        norm_ptrs = norm_ptr + batch_offsets * stride_norm_batch
        tl.atomic_add(norm_ptrs, local_sum, mask=batch_mask)

@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    output_ptr,
    batch_size,
    dim,
    epsilon,
    stride_x_batch,
    stride_x_dim,
    stride_norm_batch,
    stride_out_batch,
    stride_out_dim,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Normalize input by computed L2 norms."""
    # 1D launch grid
    pid = tl.program_id(0)
    
    # Calculate batch and dimension indices from single pid
    num_batch_blocks = tl.cdiv(batch_size, BLOCK_B)
    batch_block_idx = pid // num_batch_blocks
    dim_block_idx = pid % num_batch_blocks
    
    # Batch offsets for this block
    batch_start = batch_block_idx * BLOCK_B
    batch_offsets = batch_start + tl.arange(0, BLOCK_B)
    batch_mask = batch_offsets < batch_size
    
    # Dimension offsets for this block
    dim_start = dim_block_idx * BLOCK_D
    dim_offsets = dim_start + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < dim
    
    # Load and compute inverse norm
    norm_ptrs = norm_ptr + batch_offsets * stride_norm_batch
    norm_values = tl.load(norm_ptrs, mask=batch_mask, other=0.0)
    inv_norm = 1.0 / (tl.sqrt(norm_values + epsilon))
    
    # Input pointers
    x_ptrs = x_ptr + (
        batch_offsets[:, None] * stride_x_batch +
        dim_offsets[None, :] * stride_x_dim
    )
    
    # Output pointers
    out_ptrs = output_ptr + (
        batch_offsets[:, None] * stride_out_batch +
        dim_offsets[None, :] * stride_out_dim
    )
    
    # Load, normalize, and store
    mask = batch_mask[:, None] & dim_mask[None, :]
    x_block = tl.load(x_ptrs, mask=mask, other=0.0)
    normalized = x_block * inv_norm[:, None]
    tl.store(out_ptrs, normalized, mask=mask)

def triton_l2_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dim() == 2, "Input must have exactly 2 dimensions (batch, dim)"
    
    batch_size, dim = x.shape
    output = torch.empty_like(x)
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    output = output.contiguous()
    
    stride_x_batch = x.stride(0)
    stride_x_dim = x.stride(1)
    stride_out_batch = output.stride(0)
    stride_out_dim = output.stride(1)
    
    # Allocate norm tensor for accumulation
    norm_sums = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
    
    # Optimized block sizes
    BLOCK_B = 128
    BLOCK_D = 256
    
    # Calculate grid dimensions - 1D grid
    grid = (triton.cdiv(batch_size, BLOCK_B) * triton.cdiv(dim, BLOCK_D),)
    
    # Launch norm computation kernel
    l2_norm_kernel[grid](
        x,
        norm_sums,
        batch_size,
        dim,
        stride_x_batch,
        stride_x_dim,
        norm_sums.stride(0),
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )
    
    # Launch normalization kernel with same grid
    normalize_kernel[grid](
        x,
        norm_sums,
        output,
        batch_size,
        dim,
        epsilon,
        stride_x_batch,
        stride_x_dim,
        norm_sums.stride(0),
        stride_out_batch,
        stride_out_dim,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )
    
    return output

class ModelNew(nn.Module):
    """L2 normalization layer with optimized Triton kernels."""
    def __init__(self, epsilon: float = 1e-8):
        super(ModelNew, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle multi-dimensional tensors by flattening to 2D
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            result = triton_l2_norm(x, self.epsilon)
            return result.view(original_shape)
        return triton_l2_norm(x, self.epsilon)
