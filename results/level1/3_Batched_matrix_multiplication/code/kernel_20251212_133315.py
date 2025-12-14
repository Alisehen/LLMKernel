import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Original configs with adjusted num_warps for better occupancy
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        
        # New configs with higher num_warps for better SM utilization
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        
        # Configs with increased num_stages for better L1/L2 cache behavior
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'batch_size'],
)
@triton.jit
def bmm_kernel_optimized(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    batch_size,
    # Strides
    stride_Ab, stride_Am, stride_Ak,
    stride_Bb, stride_Bk, stride_Bn,
    stride_Cb, stride_Cm, stride_Cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized 3D grid BMM kernel with autotuned tile sizes for optimal occupancy.
    """
    # -----------------------------------------------------------
    # Map 3D program ids to tile coordinates
    # -----------------------------------------------------------
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)
    
    # Group M dimension for better L2 cache locality
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    group_id = pid_m // GROUP_SIZE_M
    group_size = tl.minimum(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m_in_group = pid_m % GROUP_SIZE_M
    pid_m = group_id * GROUP_SIZE_M + pid_m_in_group
    
    # -----------------------------------------------------------
    # Create block pointers with precomputed offsets
    # -----------------------------------------------------------
    # A block pointer
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    
    # B block pointer  
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    
    # Precompute base pointers for this batch
    A_batch_ptr = A_ptr + pid_batch * stride_Ab
    B_batch_ptr = B_ptr + pid_batch * stride_Bb
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Main computation loop
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute mask for A
        a_mask = (offs_am[:, None] < M) & ((k * BLOCK_SIZE_K + offs_ak[None, :]) < K)
        
        # Load A tile
        A_ptrs = A_batch_ptr + (offs_am[:, None] * stride_Am + 
                               (k * BLOCK_SIZE_K + offs_ak[None, :]) * stride_Ak)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        
        # Compute mask for B
        b_mask = ((k * BLOCK_SIZE_K + offs_bk[:, None]) < K) & (offs_bn[None, :] < N)
        
        # Load B tile
        B_ptrs = B_batch_ptr + ((k * BLOCK_SIZE_K + offs_bk[:, None]) * stride_Bk + 
                                offs_bn[None, :] * stride_Bn)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiplication with Tensor Cores
        accumulator += tl.dot(a, b, allow_tf32=True)
    
    # -----------------------------------------------------------
    # Write back results
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    C_batch_ptr = C_ptr + pid_batch * stride_Cb
    C_ptrs = C_batch_ptr + (offs_cm[:, None] * stride_Cm + offs_cn[None, :] * stride_Cn)
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, accumulator, mask=c_mask)

def triton_bmm_optimized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized batched matrix multiplication with 3D grid and autotuning.
    """
    assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3D tensors"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, M, K = A.shape
    _, _, N = B.shape
    
    # Allocate output tensor
    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    
    # 3D grid for maximum parallelism: (M_tiles, N_tiles, batch)
    grid = (
        triton.cdiv(M, 64),  # Use smaller default tile for autotune start
        triton.cdiv(N, 64),
        batch_size
    )
    
    # Launch optimized kernel with autotuning
    bmm_kernel_optimized[grid](
        A, B, C,
        M, N, K,
        batch_size,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication model using 3D grid Triton kernels with autotuning.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication with optimized Triton kernel.
        
        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).
            
        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_bmm_optimized(A, B)
