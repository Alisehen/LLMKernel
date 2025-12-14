import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N, M, K, L,
    stride_An, stride_Am, stride_Ak,
    stride_Bk, stride_Bl,
    stride_Cn, stride_Cm, stride_Cl,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        
        # Compute A pointer with proper indexing
        A_block_ptr = (
            A_ptr + 
            pid_batch * stride_An +
            offs_m[:, None] * stride_Am + 
            (k_offset + offs_k[None, :]) * stride_Ak
        )
        
        # Compute B pointer with proper indexing  
        B_block_ptr = (
            B_ptr +
            (k_offset + offs_k[:, None]) * stride_Bk +
            offs_n[None, :] * stride_Bl
        )
        
        # Load A and B with masking for boundaries
        a = tl.load(
            A_block_ptr,
            mask=(offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K),
            other=0.0
        )
        b = tl.load(
            B_block_ptr,
            mask=((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < L),
            other=0.0
        )
        
        # Compute matrix multiplication
        if USE_TENSOR_CORES and ACC_TYPE == tl.float32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
    
    # Store result to C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    C_ptr_block = (
        C_ptr +
        pid_batch * stride_Cn +
        offs_cm[:, None] * stride_Cm + 
        offs_cn[None, :] * stride_Cl
    )
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(C_ptr_block, acc, mask=c_mask)


@triton.jit
def batched_matmul_kernel_optimized(
    A_ptr,
    B_ptr,
    C_ptr,
    N, M, K, L,
    stride_An, stride_Am, stride_Ak,
    stride_Bk, stride_Bl,
    stride_Cn, stride_Cm, stride_Cl,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(L, BLOCK_N)
    
    # Reconstruct pid_m and pid_n from flattened pid
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        
        # Compute A pointer
        A_block_ptr = (
            A_ptr + 
            pid_batch * stride_An +
            offs_m[:, None] * stride_Am + 
            (k_offset + offs_k[None, :]) * stride_Ak
        )
        
        # Compute B pointer
        B_block_ptr = (
            B_ptr +
            (k_offset + offs_k[:, None]) * stride_Bk +
            offs_n[None, :] * stride_Bl
        )
        
        # Load A and B with masking for boundaries
        a = tl.load(
            A_block_ptr,
            mask=(offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K),
            other=0.0
        )
        b = tl.load(
            B_block_ptr,
            mask=((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < L),
            other=0.0
        )
        
        # Compute matrix multiplication
        if USE_TENSOR_CORES and ACC_TYPE == tl.float32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
    
    # Store result to C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    C_ptr_block = (
        C_ptr +
        pid_batch * stride_Cn +
        offs_cm[:, None] * stride_Cm + 
        offs_cn[None, :] * stride_Cl
    )
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(C_ptr_block, acc, mask=c_mask)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 2
    N, M, K = A.shape
    L = B.shape[1]
    
    C = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    # Determine accumulator type and whether to use tensor cores
    if A.dtype == torch.float16:
        acc_dtype = tl.float32
        use_tensor_cores = True
    elif A.dtype == torch.bfloat16:
        acc_dtype = tl.float32
        use_tensor_cores = True
    elif A.dtype == torch.float32:
        acc_dtype = tl.float32
        use_tensor_cores = True
    else:
        acc_dtype = tl.float32
        use_tensor_cores = False
    
    # Choose block sizes based on problem dimensions
    # Conservative sizes to avoid shared memory overflow
    if M <= 128 and L <= 128:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_stages = 3
        num_warps = 4
    elif M <= 256 and L <= 256:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_stages = 3
        num_warps = 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_stages = 2  # Reduced stages for larger problems
        num_warps = 8
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(L, BLOCK_N)
    total_blocks = grid_m * grid_n
    
    # Use optimized kernel for larger problems
    if total_blocks * N > 256:  # Heuristic for when to use optimized kernel
        grid_opt = (total_blocks, N)
        
        batched_matmul_kernel_optimized[grid_opt](
            A, B, C,
            N, M, K, L,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            USE_TENSOR_CORES=use_tensor_cores,
            ACC_TYPE=acc_dtype,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        grid = (N, grid_m, grid_n)
        
        batched_matmul_kernel[grid](
            A, B, C,
            N, M, K, L,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            USE_TENSOR_CORES=use_tensor_cores,
            ACC_TYPE=acc_dtype,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_batched_matmul(A, B)
