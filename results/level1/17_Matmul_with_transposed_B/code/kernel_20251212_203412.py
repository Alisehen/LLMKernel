import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Tile configuration for better cache usage
    TILE_M: tl.constexpr, TILE_N: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    # Program IDs represent the tile in the output matrix
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Compute the starting indices for this tile
    offs_m = pid_m * TILE_M * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * TILE_N * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for the first tile of A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension with tiling
    for k in range(0, K, BLOCK_K):
        # Load tiles from A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        
        # Matrix multiplication with tensor core optimization
        if USE_TF32:
            # Use tensor cores with TF32 precision
            a_f16 = a.to(tl.float16)
            b_f16 = b.to(tl.float16)
            accumulator += tl.dot(a_f16, b_f16, allow_tf32=True).to(tl.float32)
        else:
            # Use standard float32 matmul
            accumulator += tl.dot(a, b)
        
        # Move to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store the result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

@triton.jit
def transpose_kernel(
    src_ptr, dst_ptr,
    M, N,
    stride_src_m, stride_src_n,
    stride_dst_n, stride_dst_m,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.cdiv(M * N, BLOCK_SIZE)
    
    for i in range(0, num_blocks, BLOCK_SIZE):
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + i
        
        # Compute original coordinates
        m = idx // N
        n = idx % N
        mask = idx < M * N
        
        # Load with original coordinates
        val = tl.load(src_ptr + m * stride_src_m + n * stride_src_n, mask=mask)
        
        # Store with transposed coordinates
        tl.store(dst_ptr + n * stride_dst_n + m * stride_dst_m, val, mask=mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Input validation
    assert a.dim() == 2 and b.dim() == 2
    assert a.size(1) == b.size(1), f"K dimension mismatch: {a.size(1)} != {b.size(1)}"
    
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b
    
    # Transpose B to get B^T
    b_t = torch.empty(N, K, device=b.device, dtype=b.dtype)
    
    # Launch transpose kernel
    BLOCK_SIZE_TRANSPOSE = 1024
    grid_transpose = lambda meta: (triton.cdiv(M * N, meta['BLOCK_SIZE']),)
    transpose_kernel[grid_transpose](
        b, b_t,
        N, K,  # Note: we transpose (N, K) to (K, N) but store as (N, K)
        b.stride(0), b.stride(1),
        b_t.stride(0), b_t.stride(1),
        BLOCK_SIZE=BLOCK_SIZE_TRANSPOSE
    )
    
    # Now compute C = A * B^T (where B^T is b_t.T, but b_t is already transposed layout)
    # Actually, we need to compute A @ b_t.T, but b_t is (N, K) so b_t.T is (K, N)
    # So we need to compute A (M, K) @ b_t.T (K, N)
    # In our kernel, we're computing A @ B where B is (K, N)
    
    # Create output tensor
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    
    # Configure kernel parameters for optimal performance
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    TILE_M, TILE_N = 4, 4
    
    # Check if we can use TF32 tensor cores
    USE_TF32 = a.dtype == torch.float32 and b.dtype == torch.float32
    
    # Compute grid size
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M'] * meta['TILE_M']),
        triton.cdiv(N, meta['BLOCK_N'] * meta['TILE_N']),
    )
    
    # Launch matmul kernel
    matmul_kernel[grid](
        a, b_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),  # b_t is (N, K) but we want (K, N) access pattern
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        USE_TF32=USE_TF32,
    )
    
    return c

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication: C = A * B.T
        
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul(A, B)
