import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triu_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized upper triangular matrix multiplication kernel.
    Computes C = triu(A * B) for upper triangular matrices A and B.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for rows and columns
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Boundary masks
    m_mask = offs_m < N
    n_mask = offs_n < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # For upper triangular matrices, only need to compute where row <= column
    # and only iterate over valid k range for these rows/columns
    start_k = tl.maximum(pid_m * BLOCK_M, pid_n * BLOCK_N)
    
    # Iterate over K dimension in blocks
    for k_block in range(start_k, N, BLOCK_K):
        k_offs = k_block + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N
        
        # Load A block: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B block: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Only accumulate where k is between row and column for upper triangular matrices
        # For A[i,k] we need i <= k, for B[k,j] we need k <= j
        # So we need k such that max(i, start_k) <= k <= min(j, N-1)
        # We enforce this with masks
        
        # Create broadcastable indices
        i_idx = offs_m[:, None, None]  # BLOCK_M, 1, 1
        j_idx = offs_n[None, None, :]  # 1, 1, BLOCK_N
        k_idx = k_offs[None, :, None]  # 1, BLOCK_K, 1
        
        # Upper triangular conditions
        valid_a = i_idx <= k_idx  # A[i,k] non-zero if i <= k
        valid_b = k_idx <= j_idx  # B[k,j] non-zero if k <= j
        
        # Combine conditions
        valid = valid_a & valid_b
        
        # Apply conditions by zeroing invalid elements
        a_expanded = tl.where(valid_a[:, :, 0], a, 0.0)
        b_expanded = tl.where(valid_b[0, :, :], b, 0.0)
        
        # Matrix multiplication
        acc += tl.dot(a_expanded, b_expanded, allow_tf32=True)
    
    # Apply final upper triangular condition (row <= column)
    row_cond = offs_m[:, None] <= offs_n[None, :]
    acc = tl.where(row_cond, acc, 0.0)
    
    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

def triton_triu_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized Triton implementation of upper triangular matrix multiplication.
    """
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape == B.shape
    N = A.shape[0]
    
    # Allocate output
    C = torch.zeros((N, N), device=A.device, dtype=A.dtype)
    
    # Autotuning configurations
    configs = []
    for num_stages in [3, 4]:
        for BLOCK_M in [64, 128]:
            for BLOCK_N in [64, 128]:
                for BLOCK_K in [32, 64, 128]:
                    for num_warps in [4, 8]:
                        if BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N <= 49152:
                            configs.append(triton.Config({
                                'BLOCK_M': BLOCK_M,
                                'BLOCK_N': BLOCK_N,
                                'BLOCK_K': BLOCK_K,
                            }, num_stages=num_stages, num_warps=num_warps))
    
    @triton.autotune(configs=configs, key=['N'])
    @triton.jit
    def kernel_wrapper(
        A_ptr, B_ptr, C_ptr, N,
        stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        triu_matmul_kernel(
            A_ptr, B_ptr, C_ptr, N,
            stride_am, stride_ak, stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
    
    # Launch kernel - use fixed block size for grid computation
    grid = (triton.cdiv(N, 64), triton.cdiv(N, 64))
    kernel_wrapper[grid](
        A, B, C, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C

class ModelNew(nn.Module):
    """
    Optimized model for upper triangular matrix multiplication using Triton.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication for upper triangular matrices.
        
        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).
            
        Returns:
            torch.Tensor: The product of A and B, also upper triangular.
        """
        return triton_triu_matmul(A, B)
