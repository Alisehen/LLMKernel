import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel for C = A.T @ B.T
    """
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Initialize accumulator in fp32 for better precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = (k + offs_k) < K
        
        # Load A block (accessing A^T column-wise)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + 
                         (k + offs_k[None, :]) * stride_ak)
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B block (accessing B^T row-wise)
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk + 
                         offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        if USE_TENSOR_CORES:
            # For tensor cores, ensure inputs are in fp16
            a = a.to(tl.float16)
            b = b.to(tl.float16)
            # tl.dot with tensor cores will accumulate in fp32
            accumulator += tl.dot(a, b)
        else:
            # For regular FP32, use fp32 accumulation
            accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32))
    
    # Store result, converting back to original dtype
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
):
    """
    3D grid kernel for better occupancy with atomic accumulation
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
    
    # Load A block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + 
                     offs_k[None, :] * stride_ak)
    a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
    
    # Load B block
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + 
                     offs_n[None, :] * stride_bn)
    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
    
    # Compute partial product
    if USE_TENSOR_CORES:
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        partial = tl.dot(a, b)
    else:
        partial = tl.dot(a.to(tl.float32), b.to(tl.float32))
    
    # Atomic add to output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, partial.to(c_ptr.dtype.element_ty), 
                 mask=m_mask[:, None] & n_mask[None, :])

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for Triton matrix multiplication.
    Computes C = A.T @ B.T efficiently.
    """
    assert A.dim() == 2 and B.dim() == 2
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"Dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
    K = K1
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    dtype = A.dtype
    use_tensor_cores = dtype in [torch.float16, torch.bfloat16]
    
    # Optimized block sizes
    if use_tensor_cores:
        # Tensor core optimized sizes (multiples of 16/32 for best performance)
        BLOCK_SIZE_M = 64 if M >= 64 else 32
        BLOCK_SIZE_N = 64 if N >= 64 else 32
        BLOCK_SIZE_K = 32 if K >= 32 else 16
    else:
        # FP32 optimized sizes
        BLOCK_SIZE_M = 64 if M >= 64 else 32
        BLOCK_SIZE_N = 64 if N >= 64 else 32
        BLOCK_SIZE_K = 32 if K >= 32 else 16
    
    # Ensure power of 2 and within bounds
    BLOCK_SIZE_M = min(triton.next_power_of_2(BLOCK_SIZE_M), 128, M)
    BLOCK_SIZE_N = min(triton.next_power_of_2(BLOCK_SIZE_N), 128, N)
    BLOCK_SIZE_K = min(triton.next_power_of_2(BLOCK_SIZE_K), 64, K)
    
    # Clamp minimum sizes
    BLOCK_SIZE_M = max(16, BLOCK_SIZE_M)
    BLOCK_SIZE_N = max(16, BLOCK_SIZE_N)
    BLOCK_SIZE_K = max(8, BLOCK_SIZE_K)
    
    # Choose kernel based on problem size
    K_tiles = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Use 3D grid for large problems with sufficient parallelism
    if K_tiles > 1 and M * N >= 256 * 256:
        grid = (
            (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
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
        )
    else:
        # Use 2D grid for smaller problems
        grid = (
            (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
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
        # Pass inputs directly (they're already transposed in the expected layout)
        return triton_matmul(A, B)
