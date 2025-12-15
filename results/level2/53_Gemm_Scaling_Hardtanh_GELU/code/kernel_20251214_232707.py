import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Balanced config: 32x128 = 4096 elements, 4 warps, good register pressure
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # Conservative fallback: 16x64 = 1024 elements, 2 warps, minimal registers
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
        # Higher occupancy: 64x64 = 4096 elements, 8 warps for memory-bound cases
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def fused_gemm_scale_hardtanh_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    scaling_factor, hardtanh_min, hardtanh_max,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid with program IDs for better cache locality
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    # Precompute scaling constant for GELU (keeps in register)
    sqrt_2_over_pi = 0.7978845608
    gelu_coeff = 0.044715
    half_scaling = 0.5 * scaling_factor
    
    # Offsets for the M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator to zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for loading A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Main matmul loop with software pipelining
    for k in range(0, K, BLOCK_K):
        # Precompute masks once per iteration
        k_remaining = tl.minimum(BLOCK_K, K - k)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        
        # Load tiles
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Accumulate with TF32 tensor cores
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias with fused scaling to reduce register pressure
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=(offs_n < N), other=0.0)
    acc += bias * scaling_factor
    
    # Apply hardtanh with recomputed bounds (cheap ops)
    acc_min = tl.maximum(acc, hardtanh_min)
    acc = tl.minimum(acc_min, hardtanh_max)
    
    # GELU (tanh approximation) with optimized register usage
    x = acc
    # Recompute x^2 and x^3 to avoid storing intermediates
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
    # Use optimized tanh approximation: (1 - exp(-2x))/(1 + exp(-2x))
    # Avoids computing exp twice
    exp_neg_2_inner = tl.exp(-2.0 * inner)
    tanh_inner = (1.0 - exp_neg_2_inner) / (1.0 + exp_neg_2_inner)
    acc = 0.5 * x * (1.0 + tanh_inner)
    
    # Store output with masking
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_gemm_scale_hardtanh_gelu(x, weight, bias, scaling_factor, hardtanh_min, hardtanh_max):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    M, K = x.shape
    N = weight.shape[0]
    
    # Transpose weight for column-major access
    weight_t = weight.t()
    
    # Output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Define kernel grid - 1D for better scheduling
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Launch kernel
    fused_gemm_scale_hardtanh_gelu_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor, hardtanh_min, hardtanh_max,
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
    
    def forward(self, x):
        return fused_gemm_scale_hardtanh_gelu(
            x, self.weight, self.bias, 
            self.scaling_factor, self.hardtanh_min, self.hardtanh_max
        )
