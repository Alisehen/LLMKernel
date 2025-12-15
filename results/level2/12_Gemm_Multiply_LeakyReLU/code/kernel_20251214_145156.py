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
    # Group size for better memory coalescing
    GROUP_M: tl.constexpr = 8,
    # Optional: allow TF32
    USE_TF32: tl.constexpr = True,
):
    """
    Fused kernel for: GEMM → Bias Add → Multiply → LeakyReLU
    Computes: C = LeakyReLU((A @ B^T + bias) * multiplier)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    # Group blocks for better L2 cache locality
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Pre-compute masks for A and B to avoid recomputation
    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_offset = k
        
        # Load A and B with appropriate masks
        a = tl.load(a_ptrs, mask=a_mask & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & b_mask, other=0.0)
        
        # Accumulate with tensor cores
        acc += tl.dot(a, b, allow_tf32=USE_TF32)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # ----------------------------------------------------------
    # Add bias (broadcast over rows)
    bias_ptrs = bias_ptr + offs_bn
    bias_mask = offs_bn < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]  # Broadcast bias across rows
    
    # ----------------------------------------------------------
    # Apply fused operations: Multiply → LeakyReLU
    acc = acc * multiplier
    # LeakyReLU: x if x > 0 else negative_slope * x
    acc = tl.where(acc > 0, acc, acc * negative_slope)
    
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
    bias: torch.Tensor,
    multiplier: float,
    negative_slope: float,
    configs=None
) -> torch.Tensor:
    """
    Fused GEMM + Bias Add + Multiply + LeakyReLU operation.
    
    Args:
        x: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Bias tensor of shape (N,)
        multiplier: Scalar multiplier
        negative_slope: Slope for negative values in LeakyReLU
        configs: Optional list of Triton configs for autotune
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
    
    # Define optimized autotune configs based on conditional rules
    # Multi-input fusion (3+ loads) -> prefer num_stages=2, num_warps=4
    if configs is None:
        configs = [
            # Conservative baseline: 4 warps, 2 stages
            triton.Config(
                {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                num_stages=2, num_warps=4
            ),
            # Try 8 warps for compute-bound cases (low register pressure)
            triton.Config(
                {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                num_stages=3, num_warps=8
            ),
            # Small block for high occupancy
            triton.Config(
                {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4},
                num_stages=2, num_warps=4
            ),
        ]
    
    # Grid function
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
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
        **config.kwargs
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
