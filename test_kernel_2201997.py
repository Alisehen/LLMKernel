import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    # Pointers to matrices
    A_ptr,
    B_ptr,
    C_ptr,
    # Matrix dimensions
    b, i, j, l, k,
    # Strides
    stride_A_b, stride_A_i, stride_A_j, stride_A_l,
    stride_B_l, stride_B_k,
    stride_C_b, stride_C_i, stride_C_j, stride_C_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute
    pid = tl.program_id(axis=0)
    total_blocks_m = tl.cdiv(i, BLOCK_SIZE_M)
    total_blocks_n = tl.cdiv(k, BLOCK_SIZE_N)
    
    # Batch and j dimension are flattened together
    pid_batch_j = pid // (total_blocks_m * total_blocks_n)
    pid_mn = pid % (total_blocks_m * total_blocks_n)
    pid_m = pid_mn // total_blocks_n
    pid_n = pid_mn % total_blocks_n
    
    # Batch and j offsets
    batch_idx = pid_batch_j // j
    j_idx = pid_batch_j % j
    
    # ----------------------------------------------------------
    # Create block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # -----------------------------------------------------------
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    num_k_blocks = tl.cdiv(l, BLOCK_SIZE_K)
    
    for k_idx in range(0, num_k_blocks):
        # Load A block
        a_offs_k = k_idx * BLOCK_SIZE_K + offs_k
        a_ptrs = (
            A_ptr +
            batch_idx * stride_A_b +
            offs_m[:, None] * stride_A_i +
            j_idx * stride_A_j +
            a_offs_k[None, :] * stride_A_l
        )
        
        # Load B block  
        b_ptrs = (
            B_ptr +
            a_offs_k[:, None] * stride_B_l +
            offs_n[None, :] * stride_B_k
        )
        
        # Create masks
        a_mask = (batch_idx < b) & (offs_m[:, None] < i) & (j_idx < j) & (a_offs_k[None, :] < l)
        b_mask = (a_offs_k[:, None] < l) & (offs_n[None, :] < k)
        
        # Load and accumulate
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_val = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b_val, allow_tf32=False)
    
    # -----------------------------------------------------------
    # Write back the block of C
    c_ptrs = (
        C_ptr +
        batch_idx * stride_C_b +
        offs_m[:, None] * stride_C_i +
        j_idx * stride_C_j +
        offs_n[None, :] * stride_C_k
    )
    
    c_mask = (batch_idx < b) & (offs_m[:, None] < i) & (j_idx < j) & (offs_n[None, :] < k)
    
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs 4D tensor-matrix multiplication: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    
    Args:
        A: Input 4D tensor of shape (b, i, j, l)
        B: Input matrix of shape (l, k)
    
    Returns:
        Output 4D tensor of shape (b, i, j, k)
    """
    # Get dimensions
    b, i, j, l = A.shape
    k = B.shape[1]
    
    # Allocate output tensor
    C = torch.empty((b, i, j, k), device=A.device, dtype=A.dtype)
    
    # Ensure tensors are contiguous
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # Choose optimal block sizes for the hardware
    # Reduced block sizes to avoid shared memory overflow
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    grid_batch_j = b * j
    grid_m = triton.cdiv(i, BLOCK_SIZE_M)
    grid_n = triton.cdiv(k, BLOCK_SIZE_N)
    
    # Use 1D grid
    grid = (grid_batch_j * grid_m * grid_n,)
    
    # Launch kernel with reduced number of stages to conserve shared memory
    batched_matmul_kernel[grid](
        A, B, C,
        b, i, j, l, k,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,
        num_stages=2,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Performs 4D tensor-matrix multiplication: 
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs the 4D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
            B (torch.Tensor): Input matrix of shape (l, k)

        Returns:
            torch.Tensor: Output 4D tensor of shape (b, i, j, k)
        """
        return triton_batched_matmul(A, B)
