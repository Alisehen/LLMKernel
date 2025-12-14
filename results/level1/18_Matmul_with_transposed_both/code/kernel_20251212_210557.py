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
    USE_FP32_ACC: tl.constexpr,
):
    """
    Optimized matrix multiplication kernel for C = A.T @ B.T
    A: (K, M) in memory, accessed as (M, K) for A.T
    B: (N, K) in memory, accessed as (K, N) for B.T
    """
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the current block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Masks for boundary checks
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Initialize accumulator
    if USE_FP32_ACC:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    
    # Pointer increments
    a_ptr += offs_m[:, None] * stride_am
    b_ptr += offs_n[None, :] * stride_bn
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute k offsets for this block
        k_offs = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offs < K
        
        # Load A block (access A.T which is shape (M, K))
        # A is stored as (K, M), so A.T[m, k] = A[k, m]
        a_ptrs = a_ptr + (k_offs[None, :] * stride_ak)
        a = tl.load(a_ptrs, 
                    mask=m_mask[:, None] & k_mask[None, :], 
                    other=0.0)
        
        # Load B block (access B.T which is shape (K, N))
        # B is stored as (N, K), so B.T[k, n] = B[n, k]
        b_ptrs = b_ptr + (k_offs[:, None] * stride_bk)
        b = tl.load(b_ptrs,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0)
        
        if USE_TENSOR_CORES:
            # For tensor cores, convert to fp16 if needed
            a_fp16 = a.to(tl.float16)
            b_fp16 = b.to(tl.float16)
            if USE_FP32_ACC:
                accumulator += tl.dot(a_fp16, b_fp16, acc_dtype=tl.float32)
            else:
                accumulator += tl.dot(a_fp16, b_fp16)
        else:
            # For regular FP32 computation
            a_fp32 = a.to(tl.float32)
            b_fp32 = b.to(tl.float32)
            accumulator += tl.dot(a_fp32, b_fp32)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), 
             mask=m_mask[:, None] & n_mask[None, :])

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for Triton matrix multiplication.
    Computes C = A.T @ B.T efficiently.
    
    Input shapes:
    - A: (K, M)
    - B: (N, K)
    Output shape: (M, N)
    """
    assert A.dim() == 2 and B.dim() == 2
    
    # Extract dimensions
    K1, M = A.shape  # A is (K, M) in original
    N, K2 = B.shape  # B is (N, K) in original
    
    # For A.T @ B.T, we need A.T shape (M, K) and B.T shape (K, N)
    # K must match: K from A.T columns and B.T rows
    assert K1 == K2, f"Dimension mismatch: A.shape={A.shape}, B.shape={B.shape}"
    K = K1
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    dtype = A.dtype
    use_tensor_cores = dtype in [torch.float16, torch.bfloat16]
    use_fp32_acc = dtype == torch.float32
    
    # Optimize block sizes for performance
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64 if not use_tensor_cores else 32
    
    # Adjust block sizes based on problem dimensions
    BLOCK_SIZE_M = min(BLOCK_SIZE_M, M)
    BLOCK_SIZE_N = min(BLOCK_SIZE_N, N)
    BLOCK_SIZE_K = min(BLOCK_SIZE_K, K)
    
    # Ensure block sizes are powers of 2 for better performance
    BLOCK_SIZE_M = triton.next_power_of_2(BLOCK_SIZE_M)
    BLOCK_SIZE_N = triton.next_power_of_2(BLOCK_SIZE_N)
    BLOCK_SIZE_K = triton.next_power_of_2(BLOCK_SIZE_K)
    
    # Clamp minimum sizes
    BLOCK_SIZE_M = max(16, BLOCK_SIZE_M)
    BLOCK_SIZE_N = max(16, BLOCK_SIZE_N)
    BLOCK_SIZE_K = max(16, BLOCK_SIZE_K)
    
    # Calculate grid size
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    
    # Launch kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(1), A.stride(0),  # stride_am, stride_ak for A.T
        B.stride(1), B.stride(0),  # stride_bk, stride_bn for B.T
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TENSOR_CORES=use_tensor_cores,
        USE_FP32_ACC=use_fp32_acc,
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
            A: Input tensor of shape (K, M)
            B: Input tensor of shape (N, K)
            
        Returns:
            Output tensor of shape (M, N) = A.T @ B.T
        """
        return triton_matmul(A, B)
