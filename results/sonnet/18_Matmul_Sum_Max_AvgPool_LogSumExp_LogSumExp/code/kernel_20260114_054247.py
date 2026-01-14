import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 32, 'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_B': 64, 'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_K': 256}, num_warps=4),
    ],
    key=['B', 'K'],
)
@triton.jit
def fused_linear_sum_kernel(
    x_ptr, w_sum_ptr, bias_sum_ptr, out_ptr,
    B, K,
    stride_xb, stride_xk,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Batch indices this block handles
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    
    # Initialize accumulators
    acc = tl.zeros((BLOCK_B,), dtype=tl.float32)
    
    # Process K dimension in chunks
    offs_k = tl.arange(0, BLOCK_K)
    
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < K
        
        # Load w_sum chunk (broadcast across batch)
        w_vals = tl.load(w_sum_ptr + k_idx, mask=mask_k, other=0.0)
        
        # Load x values - [BLOCK_B, BLOCK_K]
        x_ptrs = x_ptr + offs_b[:, None] * stride_xb + k_idx[None, :] * stride_xk
        combined_mask = mask_b[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=combined_mask, other=0.0)
        
        # Multiply and reduce along K
        acc += tl.sum(x_vals * w_vals[None, :], axis=1)
    
    # Add bias sum (scalar)
    bias_sum = tl.load(bias_sum_ptr)
    result = acc + bias_sum
    
    # Store results
    tl.store(out_ptr + offs_b, result, mask=mask_b)


@triton.jit
def sum_weights_kernel(
    weight_ptr, bias_ptr, w_sum_ptr, bias_sum_ptr,
    out_features, in_features,
    stride_wo, stride_wi,
    BLOCK_K: tl.constexpr,
):
    # Sum weights along out_features dimension for each in_feature
    pid_k = tl.program_id(0)
    
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < in_features
    
    # Sum over out_features dimension
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    
    for o in range(out_features):
        w_ptrs = weight_ptr + o * stride_wo + offs_k * stride_wi
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += w_vals
    
    tl.store(w_sum_ptr + offs_k, acc, mask=mask_k)
    
    # First block also computes bias sum
    if pid_k == 0:
        bias_acc = tl.zeros((1,), dtype=tl.float32)
        for o in range(out_features):
            b_val = tl.load(bias_ptr + o)
            bias_acc += b_val
        tl.store(bias_sum_ptr, bias_acc)


def fused_linear_sum(x, w_sum, bias_sum):
    B, K = x.shape
    out = torch.empty((B,), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
    
    fused_linear_sum_kernel[grid](
        x, w_sum, bias_sum, out,
        B, K,
        x.stride(0), x.stride(1),
    )
    return out.unsqueeze(1)


def compute_weight_sums(weight, bias, w_sum, bias_sum):
    out_features, in_features = weight.shape
    BLOCK_K = 256
    
    grid = (triton.cdiv(in_features, BLOCK_K),)
    
    sum_weights_kernel[grid](
        weight, bias, w_sum, bias_sum,
        out_features, in_features,
        weight.stride(0), weight.stride(1),
        BLOCK_K=BLOCK_K,
    )


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('w_sum', torch.zeros(in_features))
        self.register_buffer('bias_sum_tensor', torch.zeros(1))
        self._weights_dirty = True
        
    def forward(self, x):
        # Compute w_sum and bias_sum using optimized kernel
        # For inference, this could be cached
        w_sum = self.linear.weight.sum(dim=0)
        bias_sum = self.linear.bias.sum().view(1)
        
        # Fused kernel: x @ w_sum + bias_sum
        result = fused_linear_sum(x, w_sum, bias_sum)
        
        return result
