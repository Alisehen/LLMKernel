import torch
import torch.nn as nn
import triton
import triton.language as tl

# ==================== OPTIMIZED MATMUL + BIAS + SIGMOID KERNEL ====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_sigmoid_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), 
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    output = 1.0 / (1.0 + tl.exp(-accumulator))
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, output, mask=mask)

# ==================== OPTIMIZED MATMUL + BIAS KERNEL ====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bn
    
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

# ==================== OPTIMIZED LOGSUMEXP KERNEL ====================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 2048}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def logsumexp_kernel(
    x_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for n in range(0, N, BLOCK_N):
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (n + offs_n[None, :]) * stride_xn
        mask = (offs_m[:, None] < M) & ((n + offs_n[None, :]) < N)
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        
        # Compute block max
        block_max = tl.max(x, axis=1)
        new_max = tl.maximum(row_max, block_max)
        
        # Compute exponentials with proper numerical stability
        exp_x = tl.exp(x - new_max[:, None])
        block_sum = tl.sum(exp_x, axis=1)
        
        # Update running sum with proper scaling
        scale = tl.exp(row_max - new_max)
        row_sum = row_sum * scale + block_sum
        
        # Update max
        row_max = new_max
    
    # Final result
    result = tl.log(row_sum) + row_max
    
    output_ptrs = output_ptr + offs_m * stride_om
    tl.store(output_ptrs, result, mask=offs_m < M)

# ==================== OPTIMIZED WRAPPER FUNCTIONS ====================
def triton_linear_sigmoid(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    matmul_bias_sigmoid_kernel[grid](
        x, weight.T, output, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        output.stride(0), output.stride(1),
    )
    
    return output

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    matmul_bias_kernel[grid](
        x, weight.T, output, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        output.stride(0), output.stride(1),
    )
    
    return output

def triton_logsumexp(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise ValueError("Only dim=1 is supported")
    
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)
    
    logsumexp_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0),
    )
    
    return output

# ==================== OPTIMIZED MODEL ====================
class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = triton_linear_sigmoid(x, self.linear1.weight, self.linear1.bias)
        x = triton_linear(x, self.linear2.weight, self.linear2.bias)
        x = triton_logsumexp(x, dim=1)
        return x
