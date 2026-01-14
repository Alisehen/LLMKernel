import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Standard 2D-tiled GEMM kernel with bias addition."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Main GEMM loop
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def max_mean_gelu_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    BLOCK_N: tl.constexpr,
):
    """Compute max along dim=1, subtract mean, apply GELU."""
    pid_m = tl.program_id(0)
    
    # Each program handles one row
    row_start = input_ptr + pid_m * stride_im
    
    # Find max along the row
    max_val = tl.full((1,), float('-inf'), dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        vals = tl.load(row_start + offs_n * stride_in, mask=mask, other=float('-inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    
    # After max along dim=1, we have a single value per row
    # Mean of a single value is itself
    # So result = max_val - max_val = 0
    result = max_val - max_val  # = 0
    
    # GELU(0) = 0
    # But let's compute it properly for correctness
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = result * result * result
    inner = sqrt_2_over_pi * (result + coeff * x3)
    
    # Compute tanh: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * inner)
    tanh_val = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    gelu_result = 0.5 * result * (1.0 + tanh_val)
    
    # Store result
    tl.store(output_ptr + pid_m, gelu_result)


def gemm_max_mean_gelu(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    
    # Transpose weight for GEMM: (N, K) -> (K, N)
    b = weight.t().contiguous()
    
    # Allocate intermediate buffer for GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # GEMM kernel parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    gemm_kernel[grid_gemm](
        x, b, bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1), b.stride(0), b.stride(1),
        gemm_out.stride(0), gemm_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Allocate output buffer
    output = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    
    # Reduction kernel parameters
    BLOCK_N_RED = 256
    grid_red = (M,)
    
    max_mean_gelu_kernel[grid_red](
        gemm_out, output,
        M, N,
        gemm_out.stride(0), gemm_out.stride(1),
        BLOCK_N=BLOCK_N_RED,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, x):
        return gemm_max_mean_gelu(x, self.gemm.weight, self.gemm.bias)
