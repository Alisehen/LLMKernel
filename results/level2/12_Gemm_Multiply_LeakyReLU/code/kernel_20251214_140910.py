import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gemm_mul_leaky_relu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
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
    # Optimization parameters
    GROUP_M: tl.constexpr = 8,
    USE_TF32: tl.constexpr = True,
):
    """
    Fused kernel for: GEMM → Bias Add → Multiply → LeakyReLU
    Computes: C = LeakyReLU((A @ B^T + bias) * multiplier)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Using 2D grid for better SM occupancy
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # ----------------------------------------------------------
    # Create block pointers with swizzling for better L2 cache utilization
    # Use group ordering to improve memory coalescing
    num_m_blocks = tl.cdiv(M, BLOCK_M)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    
    # Reorder blocks for better L2 locality
    group_m = pid_m // GROUP_M
    group_m_size = min(GROUP_M, num_m_blocks - group_m * GROUP_M)
    pid_in_group = pid_m % GROUP_M
    offs_m = (group_m * GROUP_M + pid_in_group) * BLOCK_M
    
    # ----------------------------------------------------------
    # Create block pointers
    offs_am = offs_m + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # ----------------------------------------------------------
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # ----------------------------------------------------------
    # Pre-compute masks for A and B to reduce overhead in loop
    a_row_mask = offs_am < M
    b_col_mask = offs_bn < N
    
    # ----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # Use software pipelining with prefetching
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_valid = k_remaining > 0
        
        if k_valid:
            # Load next block of A and B with prefetching
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + 
                             (offs_k[None, :] + k) * stride_ak)
            b_ptrs = b_ptr + ((offs_k[:, None] + k) * stride_bk + 
                             offs_bn[None, :] * stride_bn)
            
            k_mask = offs_k < k_remaining
            a_mask = a_row_mask[:, None] & k_mask[None, :]
            b_mask = k_mask[:, None] & b_col_mask[None, :]
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # Accumulate with TF32 tensor cores
            acc += tl.dot(a, b, allow_tf32=USE_TF32)
    
    # ----------------------------------------------------------
    # Add bias (broadcast over rows)
    bias_ptrs = bias_ptr + offs_bn
    bias = tl.load(bias_ptrs, mask=b_col_mask, other=0.0)
    acc = acc + bias[None, :]  # Broadcast bias across rows
    
    # ----------------------------------------------------------
    # Apply fused operations: Multiply → LeakyReLU
    # Use fast approximations where possible
    acc = acc * multiplier
    
    # Optimized LeakyReLU using bit operations for max/min
    zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_negative = acc < zero
    # Compute leaky_relu = x * (1 - negative_slope) + max(x, 0) * negative_slope
    # This reduces operations compared to separate min/max
    neg_part = acc * negative_slope
    acc = tl.where(is_negative, neg_part, acc)
    
    # ----------------------------------------------------------
    # Write back the block of C
    # Reuse the same offsets for fusion consistency
    offs_cm = offs_am
    offs_cn = offs_bn
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + 
                      offs_cn[None, :] * stride_cn)
    c_mask = a_row_mask[:, None] & b_col_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_gemm_mul_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float,
    negative_slope: float,
    configs=None
) -> torch.Tensor:
    """
    Fused GEMM + Bias Add + Multiply + LeakyReLU operation.
    """
    # Check inputs
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"Shape mismatch: x {x.shape}, weight {weight.shape}"
    assert bias.shape == (N,), f"Bias shape mismatch: expected ({N},), got {bias.shape}"
    
    # Prepare output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight_t = weight.t().contiguous()
    bias = bias.contiguous()
    
    # Optimized configs for RTX 4090 (Ada Lovelace)
    # Focusing on higher SM occupancy and L2 cache utilization
    if configs is None:
        configs = [
            triton.Config({
                'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8
            }, num_stages=4, num_warps=8),
            triton.Config({
                'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8
            }, num_stages=3, num_warps=8),
            triton.Config({
                'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8
            }, num_stages=4, num_warps=8),
            triton.Config({
                'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8
            }, num_stages=3, num_warps=4),
        ]
    
    # 2D grid function for better SM occupancy
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) * meta['GROUP_M'],
            triton.cdiv(N, meta['BLOCK_N']),
        )
    
    # Launch kernel with best config
    config = configs[0]
    fused_gemm_mul_leaky_relu_kernel[grid](
        x, weight_t, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        multiplier,
        negative_slope,
        **config.kwargs,
        num_warps=config.num_warps,
        num_stages=config.num_stages
    )
    
    return c


class ModelNew(nn.Module):
    """
    Optimized model with fused GEMM + Bias Add + Multiply + LeakyReLU operation.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Initialize parameters properly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Use fused kernel
        return fused_gemm_mul_leaky_relu(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )
