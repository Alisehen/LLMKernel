import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A strides (M, K) layout
    stride_bk, stride_bn,  # B strides (K, N) layout
    stride_cm, stride_cn,  # C strides (M, N) layout
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel for C = A.T @ B.T
    with explicit transpose handling in memory access patterns.
    
    Args:
        a_ptr: Pointer to A matrix (M, K)
        b_ptr: Pointer to B matrix (K, N)
        c_ptr: Pointer to C matrix (M, N)
        M, N, K: Matrix dimensions
        stride_*: Strides for each dimension
        BLOCK_SIZE_*: Tile sizes (must be powers of 2)
        USE_TENSOR_CORES: Whether to use tensor core operations
        ACCUMULATOR_TYPE: Data type for accumulation (fp32 for fp16 inputs)
    """
    
    # Program ID for 2D grid (M, N blocks)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block pointers for A and B with transposed access patterns
    # We need to access A^T (K, M) and B^T (N, K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checks
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_TYPE)
    
    # Main accumulation loop
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = (k + offs_k) < K
        
        # Load block of A^T (access A column-wise)
        # A^T has shape (K, M), so we load columns of A
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + 
                         (k + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load block of B^T (access B row-wise)
        # B^T has shape (N, K), so we load rows of B
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + 
                         offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Matrix multiplication using tensor cores if available
        if USE_TENSOR_CORES:
            # For tensor cores, ensure proper data types and shapes
            a = a.to(tl.float16)
            b = b.to(tl.float16)
            
        # Accumulate using dot product
        accumulator += tl.dot(a, b, acc_dtype=ACCUMULATOR_TYPE)
    
    # Store result to C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), 
             mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def matmul_kernel_3d_grid(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    K_TILES: tl.constexpr,
):
    """
    Alternative kernel using 3D grid for better occupancy.
    Uses K-dimension in grid to expose more parallelism.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K
    
    # Load A block (column-wise for A^T)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + 
                     offs_k[None, :] * stride_ak)
    a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Load B block (row-wise for B^T)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + 
                     offs_n[None, :] * stride_bn)
    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
    
    if USE_TENSOR_CORES:
        a = a.to(tl.float16)
        b = b.to(tl.float16)
    
    # Compute partial product
    accumulator = tl.dot(a, b, acc_dtype=ACCUMULATOR_TYPE)
    
    # Atomic add to output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    if K_TILES > 1:
        tl.atomic_add(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), 
                     mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), 
                mask=m_mask[:, None] & n_mask[None, :])

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for Triton matrix multiplication.
    Computes C = A.T @ B.T efficiently.
    """
    # Check input dimensions
    assert A.dim() == 2 and B.dim() == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
    K = K1
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Determine optimal kernel configuration
    dtype = A.dtype
    use_tensor_cores = dtype in [torch.float16, torch.bfloat16]
    accumulator_type = tl.float32 if dtype in [torch.float16, torch.bfloat16] else tl.float32
    
    # Choose block sizes based on hardware and data type
    if use_tensor_cores:
        # Tensor cores prefer multiples of 16 for M and N, 8 for K
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
    else:
        # Regular FP32 matmul
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
    
    # Adjust block sizes to fit within tensor dimensions
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, triton.next_power_of_2(M))
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, triton.next_power_of_2(N))
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, triton.next_power_of_2(K))
    
    # Ensure block sizes are powers of 2
    BLOCK_SIZE_M = triton.next_power_of_2(BLOCK_SIZE_M)
    BLOCK_SIZE_N = triton.next_power_of_2(BLOCK_SIZE_N)
    BLOCK_SIZE_K = triton.next_power_of_2(BLOCK_SIZE_K)
    
    # Clamp to valid ranges
    BLOCK_SIZE_M = max(16, min(1024, BLOCK_SIZE_M))
    BLOCK_SIZE_N = max(16, min(1024, BLOCK_SIZE_N))
    BLOCK_SIZE_K = max(16, min(1024, BLOCK_SIZE_K))
    
    # Choose kernel based on problem size
    K_tiles = triton.cdiv(K, BLOCK_SIZE_K)
    
    if K_tiles > 4 and M * N >= 1024 * 1024:
        # Use 3D grid kernel for large problems with many K tiles
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
            K_tiles
        )
        
        matmul_kernel_3d_grid[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            USE_TENSOR_CORES=use_tensor_cores,
            ACCUMULATOR_TYPE=accumulator_type,
            K_TILES=K_tiles,
        )
    else:
        # Use 2D grid kernel for smaller problems
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )
        
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            USE_TENSOR_CORES=use_tensor_cores,
            ACCUMULATOR_TYPE=accumulator_type,
        )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A.T @ B.T)
    using high-performance Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using Triton kernels.

        Args:
            A: Input tensor of shape (K, M) [Note: swapped dimensions for transpose]
            B: Input tensor of shape (N, K) [Note: swapped dimensions for transpose]

        Returns:
            Output tensor of shape (M, N) = A.T @ B.T
        """
        return triton_matmul(A.T, B.T)  # Pass transposed inputs to kernel
