import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=16),
    ],
    key=['M', 'N', 'K']
)
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
    reciprocal,
    use_bias: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(axis=0)
    
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
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pre-compute masks for M and N boundaries
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    
    # Compute block pointers with initial K offset
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Compute masks for current K block
        k_mask = (k + offs_k[None, :]) < K
        
        # Combined masks for A and B
        a_mask = m_mask & k_mask
        b_mask = (k + offs_k[:, None]) < K
        
        # Load blocks with boundary checking
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask[:, None] & n_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Move pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Apply bias if needed
    if use_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Apply ReLU and division (via multiplication)
    acc = tl.maximum(acc, 0.0) * reciprocal
    
    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=mask)

def fused_linear_relu_div(x, weight, bias, divisor):
    # Check input dimensions
    M, K = x.shape
    N = weight.shape[0]
    
    # Ensure inputs are contiguous with optimal memory layout
    x = x.contiguous()
    weight_t = weight.t().contiguous()
    
    # Prepare output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Compute reciprocal to replace division with multiplication
    reciprocal = 1.0 / divisor
    
    # Determine USE_BIAS flag
    use_bias = bias is not None
    
    # Handle bias pointer - pass None if not using bias
    bias_ptr = bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
    
    # Grid calculation
    def grid(META):
        total_blocks = triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N'])
        return (total_blocks,)
    
    # Launch kernel with autotuner
    fused_linear_relu_div_kernel[grid](
        x, weight_t, bias_ptr, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        reciprocal,
        use_bias=use_bias,
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
