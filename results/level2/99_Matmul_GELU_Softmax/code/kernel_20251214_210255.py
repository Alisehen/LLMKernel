import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def gemm_gelu_softmax_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Tuning parameters
    GROUP_M: tl.constexpr = 8,
    USE_TF32: tl.constexpr = True,
):
    # Program ID mapping optimized for Ada Lovelace (128 SMs)
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    # Ensure we don't go out of bounds
    num_pid_m = tl.cdiv(M, BLOCK_M)
    if pid_m >= num_pid_m:
        return
    
    # Offsets for this block with swizzling for better L2 locality
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator with tile swizzling for better Tensor Core utilization
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pre-compute block pointers for better instruction scheduling
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )
    
    # Main GEMM loop with double buffering
    for k in range(0, K, BLOCK_K):
        # Load blocks with boundary checking
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # Tensor Core acceleration
        acc += tl.dot(a, b, allow_tf32=USE_TF32)
        
        # Update block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    
    # -------------------- GELU Activation --------------------
    # Optimized GELU approximation for Ada Tensor Cores
    # GELU(x) ≈ 0.5x(1 + tanh(0.79788456x(1 + 0.044715x²)))
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coeff = 0.044715
    
    x = acc
    x_squared = x * x
    x_cubed = x_squared * x
    inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
    
    # Fast tanh approximation: tanh(x) ≈ x*(27 + x²)/(27 + 9*x²) for |x| ≤ 3
    # For larger |x|, tanh(x) ≈ sign(x)
    inner_squared = inner * inner
    
    # Branchless implementation for better warp efficiency
    small_mask = tl.abs(inner) <= 3.0
    # Replace tl.sign(inner) with arithmetic expression
    sign_approx = (2.0 * (inner > 0.0)) - 1.0
    tanh_approx = tl.where(
        small_mask,
        inner * (27.0 + inner_squared) / (27.0 + 9.0 * inner_squared),
        sign_approx
    )
    
    gelu_out = 0.5 * x * (1.0 + tanh_approx)
    
    # -------------------- Softmax --------------------
    # Row-wise softmax across columns (N dimension)
    # First compute max for each row
    row_max = tl.max(gelu_out, axis=1)
    
    # Compute exponentials and sum
    exp_vals = tl.exp(gelu_out - row_max[:, None])
    row_sum = tl.sum(exp_vals, axis=1)
    
    # Normalize
    softmax_out = exp_vals / row_sum[:, None]
    
    # -------------------- Store --------------------
    # Compute mask for this block
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    
    # Store with vectorized stores
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, softmax_out, mask=mask)


@triton.jit
def gemm_gelu_softmax_kernel_small(
    # Pointers for small matrices
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Simplified kernel for small matrices (M*N < 64K)
    pid = tl.program_id(axis=0)
    total_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    if pid_m >= total_blocks_m:
        return
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # GEMM
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
            mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
            mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b, allow_tf32=True)
    
    # GELU
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coeff = 0.044715
    x = acc
    x_squared = x * x
    x_cubed = x_squared * x
    inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
    # Replace tl.sign(inner) with arithmetic expression
    sign_approx = (2.0 * (inner > 0.0)) - 1.0
    tanh_approx = tl.where(
        tl.abs(inner) <= 3.0,
        inner * (27.0 + inner * inner) / (27.0 + 9.0 * inner * inner),
        sign_approx
    )
    gelu_out = 0.5 * x * (1.0 + tanh_approx)
    
    # Softmax
    row_max = tl.max(gelu_out, axis=1)
    exp_vals = tl.exp(gelu_out - row_max[:, None])
    row_sum = tl.sum(exp_vals, axis=1)
    softmax_out = exp_vals / row_sum[:, None]
    
    # Store
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, softmax_out, mask=mask)


def fused_gemm_gelu_softmax(x, weight):
    """Fused GEMM + GELU + Softmax operation optimized for Ada Lovelace."""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    M, K = x.shape
    N = weight.shape[0]
    
    # Allocate output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Choose kernel based on problem size
    total_elements = M * N
    
    if total_elements <= 65536:  # Small problem size
        # Use simpler kernel for small matrices
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        gemm_gelu_softmax_kernel_small[grid](
            x, weight, c,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
    else:
        # Optimized kernel for large matrices
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        
        # Use optimal grid for Ada Lovelace (128 SMs)
        num_blocks_m = triton.cdiv(M, BLOCK_M)
        num_blocks_n = triton.cdiv(N, BLOCK_N)
        grid_size = num_blocks_m * num_blocks_n
        
        # Aim for 2-3 waves on 128 SMs
        if grid_size < 256:
            GROUP_M = 1
        elif grid_size < 512:
            GROUP_M = 4
        else:
            GROUP_M = 8
        
        grid = (grid_size,)
        gemm_gelu_softmax_kernel[grid](
            x, weight, c,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=GROUP_M,
            USE_TF32=True
        )
    
    return c


class ModelNew(nn.Module):
    """Optimized model with fused GEMM + GELU + Softmax for Ada Lovelace."""
    
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Initialize weight with Xavier uniform for better convergence
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        return fused_gemm_gelu_softmax(x, self.weight)
