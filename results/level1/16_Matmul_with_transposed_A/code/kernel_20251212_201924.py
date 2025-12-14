import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_transposed_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_ak,
    stride_am,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TF32: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Optimized matrix multiplication kernel for A.T @ B with 2D grid."""
    # Matrix multiplication with efficient shared memory usage
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointers to A (transposed: K x M) and B (K x N)
    # We want C = A.T @ B, so compute A[:, m]^T * B[:, n] over K
    A_ptr += (offs_m[None, :] * stride_am)
    B_ptr += (offs_n[:, None] * stride_bn)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K
        k_mask = (k_offs + offs_k) < K
        
        # Load A block: A[K, M] but we want A[M, K] for matmul
        # Load A[k:k+BLOCK_SIZE_K, m:m+BLOCK_SIZE_M]
        a_ptrs = A_ptr + ((k_offs + offs_k[:, None]) * stride_ak)
        a = tl.load(a_ptrs, 
                   mask=k_mask[:, None] & mask_m[None, :], 
                   other=0.0)
        
        # Load B block: B[k:k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N]
        b_ptrs = B_ptr + ((k_offs + offs_k[None, :]) * stride_bk)
        b = tl.load(b_ptrs, 
                   mask=k_mask[None, :] & mask_n[:, None], 
                   other=0.0)
        
        # Transpose a to get [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.trans(a)
        # b is already [BLOCK_SIZE_N, BLOCK_SIZE_K], transpose to [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.trans(b)
        
        # Matrix multiplication
        if USE_TF32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b, allow_tf32=False)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul_transposed(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Wrapper for optimized matrix multiplication A.T @ B."""
    M, K = A.shape[1], A.shape[0]
    N = B.shape[1]
    
    # Ensure tensors are contiguous and on correct device
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA device"
    A = A.contiguous()
    B = B.contiguous()
    
    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    # Conservative configurations to avoid shared memory overflow
    # Reduced block sizes and stages to fit within 101376 bytes shared memory
    configs = [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2, num_warps=4),
    ]
    
    # Enable TF32 for better performance on supported hardware
    USE_TF32 = True
    
    # Heuristic for selecting best configuration
    def grid(META):
        total_blocks_M = triton.cdiv(M, META['BLOCK_SIZE_M'])
        total_blocks_N = triton.cdiv(N, META['BLOCK_SIZE_N'])
        return (total_blocks_M * total_blocks_N, )
    
    # Warmup and benchmark
    best_config = None
    best_time = float('inf')
    
    for config in configs:
        try:
            # Warmup
            matmul_transposed_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                USE_TF32=USE_TF32,
                **config.kwargs,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            
            # Benchmark
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            matmul_transposed_kernel[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                USE_TF32=USE_TF32,
                **config.kwargs,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            end.record()
            torch.cuda.synchronize()
            
            elapsed = start.elapsed_time(end)
            if elapsed < best_time:
                best_time = elapsed
                best_config = config
        except Exception as e:
            # Skip configurations that cause resource errors
            continue
    
    # Use default config if no valid config found
    if best_config is None:
        best_config = configs[0]
    
    # Launch with best config
    matmul_transposed_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        USE_TF32=USE_TF32,
        **best_config.kwargs,
        num_warps=best_config.num_warps,
        num_stages=best_config.num_stages,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized model with Triton kernels for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized matrix multiplication A.T @ B using Triton kernels.
        
        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).
            
        Returns:
            Output tensor of shape (M, N).
        """
        return triton_matmul_transposed(A, B)
