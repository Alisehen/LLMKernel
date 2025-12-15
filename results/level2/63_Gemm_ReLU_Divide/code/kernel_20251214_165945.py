import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_linear_relu_div_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Operation parameters
    reciprocal,  # 1.0 / divisor
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,  # For better L2 cache usage
    USE_BIAS: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    
    # Number of program ids in M dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grouped ordering for better L2 cache hit rate
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create offset pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create block pointers for A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load blocks from A and B
        a = tl.load(a_ptrs, 
                   mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                   other=0.0)
        b = tl.load(b_ptrs,
                   mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N),
                   other=0.0)
        
        # Accumulate using dot product with TF32 precision
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias if needed (broadcast across M dimension)
    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Apply ReLU: max(0, x)
    acc = tl.maximum(acc, 0.0)
    
    # Divide by divisor (multiply by reciprocal)
    acc = acc * reciprocal
    
    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)

def fused_linear_relu_div(x, weight, bias, divisor):
    # Check input dimensions
    M, K = x.shape
    N = weight.shape[0]
    
    # Transpose weight for efficient memory access
    weight_t = weight.t().contiguous()
    
    # Prepare output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Compute reciprocal to replace division with multiplication
    reciprocal = 1.0 / divisor
    
    # Determine USE_BIAS flag
    use_bias = bias is not None
    
    # Define autotune configurations
    configs = [
        triton.Config({
            'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 
            'GROUP_M': 8
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32,
            'GROUP_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
            'GROUP_M': 8
        }, num_stages=3, num_warps=4),
    ]
    
    # Grid function
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Launch kernel
    fused_linear_relu_div_kernel[grid](
        x, weight_t, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        reciprocal,
        USE_BIAS=use_bias,
    )
    
    return c

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.divisor = divisor
    
    def forward(self, x):
        return fused_linear_relu_div(x, self.weight, self.bias, self.divisor)
