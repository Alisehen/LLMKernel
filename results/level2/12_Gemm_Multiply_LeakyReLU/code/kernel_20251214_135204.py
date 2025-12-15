import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gemm_mul_leaky_relu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Scalar parameters
    multiplier,
    negative_slope,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Optional: allow TF32
    USE_TF32: tl.constexpr = True,
):
    """
    Fused kernel for: GEMM → Multiply → LeakyReLU
    Computes: C = LeakyReLU((A @ B^T) * multiplier)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # ----------------------------------------------------------
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # ----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate masks.
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        
        # A block
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + 
                         offs_k[None, :] * stride_ak)
        a_mask = (offs_am[:, None] < M) & k_mask
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # B block (B is transposed: K x N)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + 
                         offs_bn[None, :] * stride_bn)
        b_mask = k_mask[:, None] & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, allow_tf32=USE_TF32)
    
    # ----------------------------------------------------------
    # Apply fused operations: Multiply → LeakyReLU
    acc = acc * multiplier
    # LeakyReLU: max(x, 0) + negative_slope * min(x, 0)
    acc_positive = tl.maximum(acc, 0.0)
    acc_negative = tl.minimum(acc, 0.0) * negative_slope
    acc = acc_positive + acc_negative
    
    # ----------------------------------------------------------
    # Write back the block of C
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + 
                      offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_gemm_mul_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    negative_slope: float,
    configs=None
) -> torch.Tensor:
    """
    Fused GEMM + Multiply + LeakyReLU operation.
    
    Args:
        x: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K) - will be transposed
        multiplier: Scalar multiplier
        negative_slope: Slope for negative values in LeakyReLU
        configs: Optional list of Triton configs for autotune
    """
    # Check inputs
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"Shape mismatch: x {x.shape}, weight {weight.shape}"
    
    # Prepare output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Transpose weight for efficient access
    weight_t = weight.t().contiguous()
    
    # Define autotune configs
    if configs is None:
        configs = [
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        ]
    
    # Grid function
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    # Launch kernel with config parameter - FIX: Pass constexpr arguments as keyword args
    config = configs[0]
    fused_gemm_mul_leaky_relu_kernel[grid, config](
        x, weight_t, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        multiplier,
        negative_slope,
        BLOCK_M=config.kwargs['BLOCK_M'],
        BLOCK_N=config.kwargs['BLOCK_N'],
        BLOCK_K=config.kwargs['BLOCK_K']
    )
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Multiply + LeakyReLU operation.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Initialize weight properly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use fused kernel
        return fused_gemm_mul_leaky_relu(
            x, self.weight, self.multiplier, self.negative_slope
        )
