import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_sum_kernel_v2(
    x_ptr, w_sum_ptr, bias_sum_ptr, out_ptr,
    B, K,
    stride_xb, stride_xk,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles BLOCK_B batch elements
    pid = tl.program_id(0)
    
    # Batch indices this block handles
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    
    # Initialize accumulators for each batch element
    acc = tl.zeros((BLOCK_B,), dtype=tl.float32)
    
    # Process K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        
        # Load w_sum chunk once (shared across all batch elements)
        w_vals = tl.load(w_sum_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
        
        # Load x values for all batch elements in this block
        # x_ptr[b, k] = x_ptr + b * stride_xb + k * stride_xk
        x_ptrs = x_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk
        combined_mask = mask_b[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=combined_mask, other=0.0)  # [BLOCK_B, BLOCK_K]
        
        # Multiply and accumulate
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)
    
    # Add bias sum
    bias_sum = tl.load(bias_sum_ptr)
    result = acc + bias_sum
    
    # Store results
    tl.store(out_ptr + offs_b, result, mask=mask_b)


@triton.jit
def fused_linear_sum_kernel_v3(
    x_ptr, w_sum_ptr, bias_sum_ptr, out_ptr,
    B, K,
    stride_xb, stride_xk,
    BLOCK_K: tl.constexpr,
):
    # One block per batch element, but use all threads for reduction
    pid_b = tl.program_id(0)
    
    if pid_b >= B:
        return
    
    # Each thread handles part of K
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    
    # Vectorized load and multiply
    num_chunks = tl.cdiv(K, BLOCK_K)
    for i in range(num_chunks):
        k_idx = i * BLOCK_K + offs_k
        mask = k_idx < K
        
        x_vals = tl.load(x_ptr + pid_b * stride_xb + k_idx * stride_xk, mask=mask, other=0.0)
        w_vals = tl.load(w_sum_ptr + k_idx, mask=mask, other=0.0)
        
        acc += x_vals * w_vals
    
    # Reduce
    result_sum = tl.sum(acc, axis=0)
    
    # Add bias
    bias_sum = tl.load(bias_sum_ptr)
    result = result_sum + bias_sum
    
    tl.store(out_ptr + pid_b, result)


def fused_linear_sum(x, w_sum, bias_sum):
    B, K = x.shape
    out = torch.empty((B,), device=x.device, dtype=x.dtype)
    
    # Use batched version for better memory reuse
    BLOCK_B = 8
    BLOCK_K = 256
    
    grid = (triton.cdiv(B, BLOCK_B),)
    
    fused_linear_sum_kernel_v2[grid](
        x, w_sum, bias_sum, out,
        B, K,
        x.stride(0), x.stride(1),
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
    )
    return out.unsqueeze(1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        # Pre-register buffers for w_sum and bias_sum
        self.register_buffer('w_sum', None)
        self.register_buffer('bias_sum_tensor', None)
        
    def forward(self, x):
        # Compute w_sum and bias_sum (could be cached if weights don't change)
        w_sum = self.linear.weight.sum(dim=0)  # (in_features,)
        bias_sum = self.linear.bias.sum().view(1)  # scalar as 1-element tensor
        
        # Fused kernel: x @ w_sum + bias_sum
        result = fused_linear_sum(x, w_sum, bias_sum)
        
        return result
