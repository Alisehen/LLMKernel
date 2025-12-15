import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gemm_gelu_softmax_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Tile for softmax reduction
    TILE_N: tl.constexpr,
):
    # 2D grid: pid_m iterates over output rows, pid_n over output columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute block of C = A * B
    a_ptr_local = a_ptr
    b_ptr_local = b_ptr
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B with masking
        a_ptrs = a_ptr_local + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr_local + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix product
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Move pointers
        a_ptr_local += BLOCK_K * stride_ak
        b_ptr_local += BLOCK_K * stride_bk
    
    # Apply GELU activation (approximation)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coeff = 0.044715
    
    x = acc
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
    # tanh approximation using exp
    tanh_inner = (tl.exp(2 * inner) - 1) / (tl.exp(2 * inner) + 1)
    gelu_out = 0.5 * x * (1 + tanh_inner)
    
    # Store GELU output to global memory
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, gelu_out, mask=mask)
    
    # Prepare for softmax reduction along columns (dim=1)
    # Each row (dim M) needs max and sum across columns (dim N)
    
    # Initialize reduction variables for each row in this block
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # Tile over N dimension for softmax reduction
    for n_start in range(0, N, TILE_N):
        n_offs = n_start + tl.arange(0, TILE_N)
        mask_tile = mask_m & (n_offs[None, :] < N)
        
        # Load tile of GELU values
        tile_ptrs = c_ptr + (offs_m[:, None] * stride_cm + n_offs[None, :] * stride_cn)
        tile_vals = tl.load(tile_ptrs, mask=mask_tile, other=-float('inf'))
        
        # Update row max
        tile_max = tl.max(tile_vals, axis=1)
        row_max = tl.maximum(row_max, tile_max)
    
    # Compute exponentials and sum
    for n_start in range(0, N, TILE_N):
        n_offs = n_start + tl.arange(0, TILE_N)
        mask_tile = mask_m & (n_offs[None, :] < N)
        
        # Load tile of GELU values
        tile_ptrs = c_ptr + (offs_m[:, None] * stride_cm + n_offs[None, :] * stride_cn)
        tile_vals = tl.load(tile_ptrs, mask=mask_tile, other=0.0)
        
        # Compute exp(x - max)
        exp_vals = tl.exp(tile_vals - row_max[:, None])
        
        # Accumulate sum
        tile_sum = tl.sum(exp_vals, axis=1)
        row_sum += tile_sum
        
        # Store intermediate exponentials
        tl.store(tile_ptrs, exp_vals, mask=mask_tile)
    
    # Normalize by sum
    for n_start in range(0, N, TILE_N):
        n_offs = n_start + tl.arange(0, TILE_N)
        mask_tile = mask_m & (n_offs[None, :] < N)
        
        # Load stored exponentials
        tile_ptrs = c_ptr + (offs_m[:, None] * stride_cm + n_offs[None, :] * stride_cn)
        exp_vals = tl.load(tile_ptrs, mask=mask_tile, other=0.0)
        
        # Normalize
        softmax_vals = exp_vals / row_sum[:, None]
        
        # Store final softmax values
        tl.store(tile_ptrs, softmax_vals, mask=mask_tile)


def fused_gemm_gelu_softmax(x, weight):
    """Fused GEMM + GELU + Softmax operation."""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    M, K = x.shape
    N = weight.shape[0]  # weight shape: [out_features, in_features]
    
    # Allocate output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Transpose weight for GEMM
    b = weight.t()
    
    # Grid for kernel launch
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    # Launch kernel with autotuned configurations
    fused_gemm_gelu_softmax_kernel[grid](
        x, b, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, TILE_N=128
    )
    
    return c


class ModelNew(nn.Module):
    """Optimized model with fused GEMM + GELU + Softmax."""
    
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return fused_gemm_gelu_softmax(x, self.weight)
