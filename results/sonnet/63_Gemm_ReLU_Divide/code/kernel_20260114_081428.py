import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_relu_div_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    divisor,
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
    
    # Double buffering: prefetch first block
    mask_k_0 = offs_k < K
    a_0 = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k_0[None, :]), other=0.0, eviction_policy='evict_last')
    b_0 = tl.load(b_ptrs, mask=(mask_k_0[:, None]) & (offs_n[None, :] < N), other=0.0, eviction_policy='evict_last')
    
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
    
    # Main loop with double buffering
    for k in range(BLOCK_K, K, BLOCK_K):
        mask_k = offs_k < K - k
        # Prefetch next iteration while computing current
        a_1 = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0, eviction_policy='evict_last')
        b_1 = tl.load(b_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0, eviction_policy='evict_last')
        
        # Compute with current buffers
        acc += tl.dot(a_0, b_0, allow_tf32=True)
        
        # Swap buffers
        a_0 = a_1
        b_0 = b_1
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Final iteration
    acc += tl.dot(a_0, b_0, allow_tf32=True)

    # Fused: add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Fused: ReLU
    acc = tl.maximum(acc, 0.0)

    # Fused: divide by constant
    acc = acc / divisor

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_linear_relu_div(x, weight, bias, divisor):
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    fused_linear_relu_div_kernel[grid](
        x, b, bias, c, M, N, K,
        divisor,
        x.stride(0), x.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.divisor = divisor

    def forward(self, x):
        return fused_linear_relu_div(x, self.weight, self.bias, self.divisor)
