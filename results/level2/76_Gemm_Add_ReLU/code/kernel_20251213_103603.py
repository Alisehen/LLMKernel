import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_relu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride variables
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Kernel for computing C = ReLU(A @ B + bias) with:
    - A: [M, K]
    - B: [K, N] (B is transposed in memory, column-major)
    - bias: [N]
    - C: [M, N]
    """
    
    # 2D grid for better parallelism
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Number of program groups along M dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_in_group = pid_m - first_pid_m
    
    # Block offsets
    offs_m = first_pid_m * BLOCK_SIZE_M + pid_m_in_group * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Pointers to blocks with precomputed offsets for better ILP
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Precompute boundary masks once for better performance
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Loop over K dimension with software pipelining
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A block with mask
        mask_a = mask_m[:, None] & (offs_k[None, :] < (K - k))
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load B block with mask
        mask_b = (offs_k[:, None] < (K - k)) & mask_n[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Compute matrix multiplication using tensor cores
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Add bias if present
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0)
        acc += bias[None, :]
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Write back result with mask
    mask_c = mask_m[:, None] & mask_n[None, :]
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul_relu(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes ReLU(a @ weight^T + bias) using Triton kernel.
    
    Args:
        a: Input tensor of shape [M, K]
        weight: Weight tensor of shape [N, K]
        bias: Bias tensor of shape [N] or None
    
    Returns:
        Output tensor of shape [M, N]
    """
    M, K = a.shape
    N, K2 = weight.shape
    assert K == K2, f"Dimension mismatch: a.shape={a.shape}, weight.shape={weight.shape}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Convert to column-major for B matrix (transpose weight)
    b = weight.t().contiguous()
    
    # Grid launch configuration - 2D grid for better parallelism
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch kernel
    matmul_relu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication, adds bias, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # Use nn.Parameter for weight to ensure it's registered correctly
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape)) if bias_shape else None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return triton_matmul_relu(x, self.weight, self.bias)
