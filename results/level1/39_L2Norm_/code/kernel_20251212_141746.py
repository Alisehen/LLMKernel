import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def l2_norm_kernel(
    x_ptr,
    output_ptr,
    batch_size,
    dim,
    stride_x_batch,
    stride_x_dim,
    stride_out_batch,
    stride_out_dim,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
    USE_FAST_SQRT: tl.constexpr = False,
):
    """
    Compute L2 norm with optimized reduction and normalization in a single kernel.
    Uses 2D block structure for efficient memory access.
    """
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    # Compute offsets
    batch_offsets = pid_batch * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    dim_offsets = pid_dim * BLOCK_SIZE_DIM + tl.arange(0, BLOCK_SIZE_DIM)
    
    # Create masks
    batch_mask = batch_offsets < batch_size
    dim_mask = dim_offsets < dim
    
    # Load block
    x_ptrs = x_ptr + (
        batch_offsets[:, None] * stride_x_batch +
        dim_offsets[None, :] * stride_x_dim
    )
    
    x_block = tl.load(x_ptrs, 
                     mask=batch_mask[:, None] & dim_mask[None, :],
                     other=0.0)
    
    # Compute local sum of squares
    local_squares = x_block * x_block
    
    # Reduce within block across dimension axis
    block_sum = tl.sum(local_squares, axis=1)
    
    # Allocate shared memory for reduction
    smem = tl.zeros((BLOCK_SIZE_BATCH,), dtype=tl.float32)
    if tl.launch_id(1) == 0:
        smem = tl.full((BLOCK_SIZE_BATCH,), 0.0, dtype=tl.float32)
    
    # Atomic add to shared memory
    tl.atomic_add(smem + tl.arange(0, BLOCK_SIZE_BATCH), block_sum)
    tl.barrier()
    
    # First batch in the block computes final norm and normalizes
    if pid_dim == 0:
        # Load accumulated sum
        total_sum = tl.load(smem + tl.arange(0, BLOCK_SIZE_BATCH))
        
        # Compute reciprocal sqrt with epsilon for numerical stability
        rsqrt_val = tl.sqrt(total_sum + 1e-8)
        inv_norm = 1.0 / rsqrt_val
        
        # Normalize the block
        normalized_block = x_block * inv_norm[:, None]
        
        # Store result
        out_ptrs = output_ptr + (
            batch_offsets[:, None] * stride_out_batch +
            dim_offsets[None, :] * stride_out_dim
        )
        tl.store(out_ptrs, normalized_block, 
                mask=batch_mask[:, None] & dim_mask[None, :])
    
    # Handle remaining dimension blocks
    if pid_dim > 0:
        # Wait for normalization to complete
        tl.barrier()
        
        # Load accumulated norm from shared memory
        total_sum = tl.load(smem + tl.arange(0, BLOCK_SIZE_BATCH))
        rsqrt_val = tl.sqrt(total_sum + 1e-8)
        inv_norm = 1.0 / rsqrt_val
        
        # Normalize this block
        normalized_block = x_block * inv_norm[:, None]
        
        # Store result
        out_ptrs = output_ptr + (
            batch_offsets[:, None] * stride_out_batch +
            dim_offsets[None, :] * stride_out_dim
        )
        tl.store(out_ptrs, normalized_block, 
                mask=batch_mask[:, None] & dim_mask[None, :])

def triton_l2_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Wrapper function for L2 normalization using Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dim() >= 2, "Input must have at least 2 dimensions"
    
    # Get input shape
    batch_size, dim = x.shape[0], x.shape[1]
    output = torch.empty_like(x)
    
    # Ensure input is contiguous
    x = x.contiguous()
    output = output.contiguous()
    
    # Get strides
    stride_x_batch = x.stride(0)
    stride_x_dim = x.stride(1) if x.dim() > 1 else 1
    
    stride_out_batch = output.stride(0)
    stride_out_dim = output.stride(1) if output.dim() > 1 else 1
    
    # Configuration - optimized for large tensors
    BLOCK_SIZE_BATCH = 64
    BLOCK_SIZE_DIM = 128
    
    # Calculate grid dimensions
    grid_batch = triton.cdiv(batch_size, BLOCK_SIZE_BATCH)
    grid_dim = triton.cdiv(dim, BLOCK_SIZE_DIM)
    
    # Launch kernel
    l2_norm_kernel[(
        grid_batch,
        grid_dim,
        1
    )](
        x,
        output,
        batch_size,
        dim,
        stride_x_batch,
        stride_x_dim,
        stride_out_batch,
        stride_out_dim,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )
    
    return output

class ModelNew(nn.Module):
    """
    L2 normalization layer with optimized Triton kernel.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Initializes the L2Norm layer.

        Args:
            epsilon (float): Small value to avoid division by zero.
        """
        super(ModelNew, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        return triton_l2_norm(x, self.epsilon)
