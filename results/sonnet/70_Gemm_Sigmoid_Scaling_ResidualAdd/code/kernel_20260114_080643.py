import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_bias_sigmoid_scale_residual_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    scaling_factor,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # GEMM loop with optimized memory access
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        
        # Simplified masking - only check boundaries once
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load bias once (single load per thread block column)
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    
    # Fused operations - all in registers, no intermediate stores
    # 1. Add bias (broadcast)
    acc = acc + bias[None, :]
    
    # 2. Keep original for residual (stays in register)
    original = acc
    
    # 3. Sigmoid: 1 / (1 + exp(-x))
    # Use fast math approximation for better performance
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sigmoid_out = 1.0 / (1.0 + exp_neg)
    
    # 4. Scale
    scaled = sigmoid_out * scaling_factor
    
    # 5. Residual add
    result = scaled + original

    # Single store for final output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=mask_c)


def fused_gemm_bias_sigmoid_scale_residual(x, weight, bias, scaling_factor):
    M, K = x.shape
    N = weight.shape[0]
    
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']), 
            triton.cdiv(N, META['BLOCK_N'])
        )
    
    fused_gemm_bias_sigmoid_scale_residual_kernel[grid](
        x, b, bias, c, M, N, K,
        scaling_factor,
        x.stride(0), x.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
    )
    
    return c


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return fused_gemm_bias_sigmoid_scale_residual(x, self.weight, self.bias, self.scaling_factor)
