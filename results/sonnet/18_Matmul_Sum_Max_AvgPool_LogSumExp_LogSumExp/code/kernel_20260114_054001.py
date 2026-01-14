import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_sum_kernel(
    x_ptr, w_sum_ptr, bias_sum_ptr, out_ptr,
    B, K,
    stride_xb, stride_xk,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one batch element
    pid_b = tl.program_id(0)
    
    # Compute dot product of x[pid_b, :] with w_sum[:]
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask = offs_k < K
        
        x_vals = tl.load(x_ptr + pid_b * stride_xb + offs_k * stride_xk, mask=mask, other=0.0)
        w_vals = tl.load(w_sum_ptr + offs_k, mask=mask, other=0.0)
        
        acc += tl.sum(x_vals * w_vals, axis=0)
    
    # Add bias sum
    bias_sum = tl.load(bias_sum_ptr)
    result = acc + bias_sum
    
    # Store result
    tl.store(out_ptr + pid_b, result)


def fused_linear_sum(x, w_sum, bias_sum):
    B, K = x.shape
    out = torch.empty((B, 1), device=x.device, dtype=x.dtype)
    
    BLOCK_K = 1024
    grid = (B,)
    
    fused_linear_sum_kernel[grid](
        x, w_sum, bias_sum, out,
        B, K,
        x.stride(0), x.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # We'll compute these in forward or register as buffers
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x):
        # Precompute the sum of weights across output dimension
        # W is (out_features, in_features), we need sum over dim=0 -> (in_features,)
        w_sum = self.linear.weight.sum(dim=0)  # (in_features,)
        bias_sum = self.linear.bias.sum().view(1)  # scalar as 1-element tensor
        
        # The result is x @ w_sum + bias_sum
        # x: (B, in_features), w_sum: (in_features,) -> result: (B,)
        result = fused_linear_sum(x, w_sum, bias_sum)
        
        return result
