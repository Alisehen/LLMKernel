import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel_2d_grid(
    x_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """GELU activation with 2D grid for better SM utilization."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    full_mask = row_mask[:, None] & col_mask[None, :]
    
    # Calculate base pointers for this block
    x_ptrs = x_ptr + (row_offsets[:, None] * n_cols + col_offsets[None, :])
    output_ptrs = output_ptr + (row_offsets[:, None] * n_cols + col_offsets[None, :])
    
    # Load input block
    x_block = tl.load(x_ptrs, mask=full_mask)
    
    # GELU computation
    # x^3 with fused multiply
    x3 = x_block * x_block * x_block
    
    # GELU constants
    a = 0.7978845608028654  # sqrt(2/π)
    b = 0.044715
    
    # t = sqrt(2/π) * (x + 0.044715 * x^3)
    t = a * tl.fma(b, x3, x_block)  # FMA: a*(b*x3 + x)
    
    # tanh(t) using exp-based formulation
    neg_2abs_t = -2.0 * tl.abs(t)
    exp_neg_2abs_t = tl.exp(neg_2abs_t)
    
    # tanh(t) = sign(t) * (1 - 2/(exp(2|t|) + 1))
    numerator = 1.0 - exp_neg_2abs_t
    denominator = 1.0 + exp_neg_2abs_t
    tanh_t = tl.where(t >= 0.0, numerator/denominator, -numerator/denominator)
    
    # GELU: 0.5 * x * (1 + tanh(t))
    output = 0.5 * x_block * tl.fma(tanh_t, 1.0, 1.0)
    
    # Store result
    tl.store(output_ptrs, output, mask=full_mask)

@triton.jit
def gelu_kernel_batched(
    x_ptr,
    output_ptr,
    batch_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Batched GELU with 3D grid for better parallelism."""
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Calculate batch offset
    batch_offset = pid_batch * batch_stride
    
    # Calculate offsets within the matrix
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks
    row_mask = row_offsets < n_rows
    col_mask = col_offsets < n_cols
    full_mask = row_mask[:, None] & col_mask[None, :]
    
    # Calculate base pointers for this block
    x_ptrs = x_ptr + batch_offset + (row_offsets[:, None] * n_cols + col_offsets[None, :])
    output_ptrs = output_ptr + batch_offset + (row_offsets[:, None] * n_cols + col_offsets[None, :])
    
    # Load input block
    x_block = tl.load(x_ptrs, mask=full_mask)
    
    # GELU computation
    x3 = x_block * x_block * x_block
    a = 0.7978845608028654
    b = 0.044715
    t = a * tl.fma(b, x3, x_block)
    neg_2abs_t = -2.0 * tl.abs(t)
    exp_neg_2abs_t = tl.exp(neg_2abs_t)
    numerator = 1.0 - exp_neg_2abs_t
    denominator = 1.0 + exp_neg_2abs_t
    tanh_t = tl.where(t >= 0.0, numerator/denominator, -numerator/denominator)
    output = 0.5 * x_block * tl.fma(tanh_t, 1.0, 1.0)
    
    # Store result
    tl.store(output_ptrs, output, mask=full_mask)

def triton_gelu_optimized(x: torch.Tensor) -> torch.Tensor:
    """Optimized Triton GELU with adaptive grid strategy."""
    output = torch.empty_like(x)
    
    # Handle different tensor shapes
    if x.dim() == 1:
        # 1D tensor: use simple 1D grid
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        @triton.jit
        def gelu_kernel_1d(
            x_ptr, output_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            x3 = x * x * x
            a = 0.7978845608028654
            b = 0.044715
            t = a * tl.fma(b, x3, x)
            neg_2abs_t = -2.0 * tl.abs(t)
            exp_neg_2abs_t = tl.exp(neg_2abs_t)
            numerator = 1.0 - exp_neg_2abs_t
            denominator = 1.0 + exp_neg_2abs_t
            tanh_t = tl.where(t >= 0.0, numerator/denominator, -numerator/denominator)
            result = 0.5 * x * tl.fma(tanh_t, 1.0, 1.0)
            tl.store(output_ptr + offsets, result, mask=mask)
        
        gelu_kernel_1d[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        
    elif x.dim() == 2:
        # 2D tensor: use 2D grid for better SM utilization
        M, N = x.shape
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        gelu_kernel_2d_grid[grid](x, output, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
        
    elif x.dim() >= 3:
        # 3D+ tensor: flatten batch dimensions for parallel processing
        # Reshape to 3D: (batch, M, N)
        if x.dim() > 3:
            batch_size = x.shape[0]
            other_dims = x.shape[1:]
            M = other_dims.numel() // other_dims[-1] if len(other_dims) > 1 else 1
            N = other_dims[-1]
            x_reshaped = x.view(batch_size, M, N)
            output_reshaped = output.view(batch_size, M, N)
        else:
            batch_size, M, N = x.shape
            x_reshaped = x
            output_reshaped = output
        
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        batch_stride = M * N
        
        grid = (batch_size, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        gelu_kernel_batched[grid](
            x_reshaped, output_reshaped, batch_stride, 
            M, N, BLOCK_SIZE_M, BLOCK_SIZE_N
        )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized GELU implementation with adaptive Triton kernels.
    Uses 2D/3D grid layouts for better SM utilization on Ada Lovelace.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_gelu_optimized(x)
