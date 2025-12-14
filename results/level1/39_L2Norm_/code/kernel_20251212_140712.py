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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute sum of squares for L2 normalization with proper reduction.
    """
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    # Optimized blocking for memory coalescing
    batch_offsets = pid_batch * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = pid_dim * BLOCK_N + tl.arange(0, BLOCK_N)
    
    batch_mask = batch_offsets < batch_size
    dim_mask = dim_offsets < dim
    
    # Coalesced loading: column-major access pattern
    x_ptrs = x_ptr + (
        dim_offsets[None, :] * stride_x_dim +
        batch_offsets[:, None] * stride_x_batch
    )
    
    # Load with broadcast mask
    x_block = tl.load(x_ptrs, mask=batch_mask[:, None] & dim_mask[None, :], other=0.0)
    
    # Warp-level reduction with higher precision
    squares = x_block * x_block
    
    # Reduce along dimension axis with tree reduction
    local_sum = tl.sum(squares, axis=1)
    
    # Atomic accumulation for each batch element with mask
    # Use tl.where style conditional instead of Python if
    for i in range(0, BLOCK_M):
        cond = batch_mask[i]
        addr = norm_ptr + batch_offsets[i]
        val = local_sum[i]
        tl.atomic_add(addr, val, mask=cond)

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
    stride_out_batch,
    stride_out_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Apply normalization using pre-computed sum of squares.
    """
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)
    
    # Optimized blocking for better cache utilization
    batch_offsets = pid_batch * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = pid_dim * BLOCK_N + tl.arange(0, BLOCK_N)
    
    batch_mask = batch_offsets < batch_size
    dim_mask = dim_offsets < dim
    
    # Load norm values (sum of squares) for the current batch block
    norm_ptrs = norm_ptr + batch_offsets
    norm_values = tl.load(norm_ptrs, mask=batch_mask, other=1.0)
    
    # Compute L2 norm with epsilon protection: sqrt(sum(x^2) + epsilon^2)
    # More numerically stable than sqrt(sum) + epsilon
    norm_values = tl.sqrt(norm_values + epsilon * epsilon)
    inv_norm = 1.0 / norm_values
    
    # Coalesced loading with broadcast
    x_ptrs = x_ptr + (
        dim_offsets[None, :] * stride_x_dim +
        batch_offsets[:, None] * stride_x_batch
    )
    x_block = tl.load(x_ptrs, mask=batch_mask[:, None] & dim_mask[None, :], other=0.0)
    
    # Normalize with broadcast
    normalized = x_block * inv_norm[:, None]
    
    # Coalesced storing
    out_ptrs = output_ptr + (
        dim_offsets[None, :] * stride_out_dim +
        batch_offsets[:, None] * stride_out_batch
    )
    tl.store(out_ptrs, normalized, mask=batch_mask[:, None] & dim_mask[None, :])

def triton_l2_norm(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dim() >= 2, "Input must have at least 2 dimensions"
    
    # Get input shape and strides
    batch_size, dim = x.shape[0], x.shape[1]
    output = torch.empty_like(x)
    
    # Ensure input is contiguous for optimal memory access
    x = x.contiguous()
    output = output.contiguous()
    
    # Get strides
    stride_x_batch = x.stride(0)
    stride_x_dim = x.stride(1)
    stride_out_batch = output.stride(0)
    stride_out_dim = output.stride(1)
    
    # Allocate tensor for sum of squares (use float32 for accumulation precision)
    norm_sums = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
    
    # Optimized block sizes for Ada Lovelace (RTX 4090)
    # BLOCK_M × BLOCK_N must be ≤ 1024 threads
    BLOCK_M = 64   # Optimized for batch dimension (warp-aligned)
    BLOCK_N = 16   # Optimized for dimension reduction
    
    # Calculate optimized grid dimensions
    grid_batch = triton.cdiv(batch_size, BLOCK_M)
    grid_dim = triton.cdiv(dim, BLOCK_N)
    
    # Launch first kernel to compute sum of squares
    l2_norm_kernel[(grid_batch, grid_dim)](
        x,
        norm_sums,
        batch_size,
        dim,
        stride_x_batch,
        stride_x_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Launch second kernel for normalization
    normalize_kernel[(grid_batch, grid_dim)](
        x,
        norm_sums,
        output,
        batch_size,
        dim,
        epsilon,
        stride_x_batch,
        stride_x_dim,
        stride_out_batch,
        stride_out_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output

class ModelNew(nn.Module):
    """
    L2 normalization layer with optimized Triton kernels for Ada Lovelace.
    """
    def __init__(self, epsilon: float = 1e-8):
        super(ModelNew, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l2_norm(x, self.epsilon)
