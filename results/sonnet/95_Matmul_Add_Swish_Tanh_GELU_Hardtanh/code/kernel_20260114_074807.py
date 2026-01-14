import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_matmul_activations_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
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
    
    # Matmul loop
    for k in range(0, K, BLOCK_K):
        mask_k = offs_k < K - k
        mask_a = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_b = (mask_k[:, None]) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Shared offsets and mask for all fused operations
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Fused: add bias (broadcast along N dimension)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Fused: Swish (silu) - x * sigmoid(x)
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sigmoid_acc = 1.0 / (1.0 + exp_neg)
    acc = sigmoid_acc * acc

    # Fused: Tanh - (exp(2x) - 1) / (exp(2x) + 1)
    acc_2 = acc + acc
    exp_2x = tl.exp(acc_2)
    exp_2x_plus_1 = exp_2x + 1.0
    acc = (exp_2x - 1.0) / exp_2x_plus_1

    # Fused: GELU - 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = acc * acc * acc
    inner = 0.7978845608028654 * (acc + 0.044715 * x_cubed)
    inner_2 = inner + inner
    exp_2inner = tl.exp(inner_2)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    acc = 0.5 * acc * (1.0 + tanh_inner)

    # Fused: Hardtanh (clamp between -1 and 1)
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Store with shared mask
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_out)


def fused_matmul_activations(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    b = weight.t().contiguous()
    
    # Reduced block sizes to fit within shared memory limits
    # RTX 4090 has 101376 bytes shared memory per SM
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']), 
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    fused_matmul_activations_kernel[grid](
        x, b, bias, c, 
        M, N, K,
        x.stride(0), x.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))

    def forward(self, x):
        return fused_matmul_activations(x, self.weight, self.add_value)
