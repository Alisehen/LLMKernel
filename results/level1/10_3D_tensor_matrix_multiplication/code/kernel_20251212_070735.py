import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Optimized for Ada Lovelace (4090): 128 SMs, focus on increasing active warps
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),  # High N for better occupancy
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),  # High M for better occupancy
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),  # Larger K for more reuse
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),   # Higher aspect ratio
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=4),    # Balanced for smaller sizes
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=6, num_warps=2),    # For small problem sizes
    ],
    key=['M', 'N', 'K']
)
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
    """Optimized batched matmul kernel with 3D grid and prefetching"""
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to the current batch
    A_batch_ptr = A_ptr + pid_batch * stride_An
    B_ptr_start = B_ptr
    C_batch_ptr = C_ptr + pid_batch * stride_Cn
    
    # Pre-compute base pointers (optimized for memory access patterns)
    a_ptrs_base = A_batch_ptr + (offs_m[:, None] * stride_Am)
    b_ptrs_base = B_ptr_start + (offs_n[None, :] * stride_Bl)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Pre-fetch first tiles
    k_offset = 0
    A_block_ptr = a_ptrs_base + (k_offset + offs_k[None, :]) * stride_Ak
    B_block_ptr = b_ptrs_base + (k_offset + offs_k[:, None]) * stride_Bk
    
    a_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
    b_mask = ((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < L)
    
    a = tl.load(A_block_ptr, mask=a_mask, other=0.0)
    b = tl.load(B_block_ptr, mask=b_mask, other=0.0)
    
    # Main computation loop with software pipelining
    for k in range(1, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        
        # Pre-fetch next tiles
        A_next_ptr = a_ptrs_base + (k_offset + offs_k[None, :]) * stride_Ak
        B_next_ptr = b_ptrs_base + (k_offset + offs_k[:, None]) * stride_Bk
        
        a_next_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        b_next_mask = ((k_offset + offs_k[:, None]) < K) & (offs_n[None, :] < L)
        
        a_next = tl.load(A_next_ptr, mask=a_next_mask, other=0.0)
        b_next = tl.load(B_next_ptr, mask=b_next_mask, other=0.0)
        
        # Compute current tile
        if USE_TENSOR_CORES and ACC_TYPE == tl.float32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
        
        # Swap for next iteration
        a, b = a_next, b_next
        a_mask, b_mask = a_next_mask, b_next_mask
    
    # Compute last tile
    if tl.cdiv(K, BLOCK_K) > 0:
        if USE_TENSOR_CORES and ACC_TYPE == tl.float32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b)
    
    # Store result with coalesced writes
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    C_ptr_block = C_batch_ptr + offs_cm[:, None] * stride_Cm + offs_cn[None, :] * stride_Cl
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(C_ptr_block, acc, mask=c_mask)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized wrapper with aggressive grid sizing for maximum occupancy"""
    assert A.dim() == 3 and B.dim() == 2
    N, M, K = A.shape
    L = B.shape[1]
    
    C = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    # Determine accumulator type and tensor core usage
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
    
    # Aggressive grid sizing for maximum occupancy on 4090 (128 SMs)
    # Target: 128 SMs × 4 warps per SM × 32 threads = 16384 active threads
    # For 3D grid: N × grid_m × grid_n blocks
    
    # Base grid size calculation
    grid_m = triton.cdiv(M, 128)  # Start with larger block size
    grid_n = triton.cdiv(L, 128)
    
    # Calculate total blocks
    total_blocks = N * grid_m * grid_n
    
    # Target minimum blocks for high occupancy on 4090
    # 128 SMs × 16 concurrent blocks per SM = 2048 blocks minimum
    # But we want more to hide memory latency
    target_min_blocks = 4096  # Double the minimum for better latency hiding
    
    # Dynamic grid adjustment for maximum occupancy
    if total_blocks < target_min_blocks:
        # Try medium blocks
        grid_m = triton.cdiv(M, 64)
        grid_n = triton.cdiv(L, 64)
        total_blocks = N * grid_m * grid_n
        
        if total_blocks < target_min_blocks:
            # Use small blocks to maximize block count
            grid_m = triton.cdiv(M, 32)
            grid_n = triton.cdiv(L, 32)
            total_blocks = N * grid_m * grid_n
            
            if total_blocks < target_min_blocks:
                # Use smallest blocks to ensure high occupancy
                grid_m = triton.cdiv(M, 16)
                grid_n = triton.cdiv(L, 16)
    
    # Launch with 3D grid
    grid = (N, grid_m, grid_n)
    
    batched_matmul_kernel[grid](
        A, B, C,
        N, M, K, L,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        USE_TENSOR_CORES=use_tensor_cores,
        ACC_TYPE=acc_dtype,
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_batched_matmul(A, B)
