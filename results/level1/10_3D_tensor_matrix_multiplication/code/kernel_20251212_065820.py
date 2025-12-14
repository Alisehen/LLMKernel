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
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_ptrs = (
        A_ptr + 
        pid_batch * stride_An +
        offs_am[:, None] * stride_Am + 
        offs_k[None, :] * stride_Ak
    )
    B_ptrs = (
        B_ptr +
        offs_k[:, None] * stride_Bk + 
        offs_bn[None, :] * stride_Bl
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k),
            other=0.0
        )
        b = tl.load(
            B_ptrs,
            mask=(offs_k[:, None] < K - k) & (offs_bn[None, :] < L),
            other=0.0
        )
        
        if USE_TENSOR_CORES:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
        
        A_ptrs += BLOCK_K * stride_Ak
        B_ptrs += BLOCK_K * stride_Bk
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    C_ptrs = (
        C_ptr +
        pid_batch * stride_Cn +
        offs_cm[:, None] * stride_Cm + 
        offs_cn[None, :] * stride_Cl
    )
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(C_ptrs, acc, mask=c_mask)


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
    GROUP_M: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(L, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs_base = A_ptr + pid_batch * stride_An + offs_am[:, None] * stride_Am
    b_ptrs_base = B_ptr + offs_bn[None, :] * stride_Bl
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        
        a_ptrs = a_ptrs_base + offs_k[None, :] * stride_Ak
        b_ptrs = b_ptrs_base + offs_k[:, None] * stride_Bk
        
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < L),
            other=0.0
        )
        
        if USE_TENSOR_CORES:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
        
        a_ptrs_base += BLOCK_K * stride_Ak
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = (
        C_ptr +
        pid_batch * stride_Cn +
        offs_cm[:, None] * stride_Cm + 
        offs_cn[None, :] * stride_Cl
    )
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 2
    N, M, K = A.shape
    L = B.shape[1]
    
    C = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    if A.dtype == torch.float16 or A.dtype == torch.bfloat16:
        acc_dtype = tl.float32
        use_tensor_cores = True
    elif A.dtype == torch.float32:
        acc_dtype = tl.float32
        use_tensor_cores = True
    else:
        acc_dtype = tl.float32
        use_tensor_cores = False
    
    if max(M, L, K) <= 512:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_stages = 3
        num_warps = 4
    elif max(M, L, K) <= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_stages = 3
        num_warps = 8
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32  # Reduced from 256x256 to avoid shared memory overflow
        num_stages = 3
        num_warps = 8
    
    grid = (
        N,
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(L, BLOCK_N)
    )
    
    if N * M * L * K > 10**7:
        GROUP_M = 8
        
        total_blocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(L, BLOCK_N)
        grid_opt = (
            total_blocks,
            N
        )
        
        batched_matmul_kernel_optimized[grid_opt](
            A, B, C,
            N, M, K, L,
            A.stride(0), A.stride(1), A.stride(2),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            USE_TENSOR_CORES=use_tensor_cores,
            ACC_TYPE=acc_dtype,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
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
