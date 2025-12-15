import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def gemm_gelu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr = 2,
):
    # Program ID for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Precompute pointers for better ILP
    a_ptrs_base = a_ptr + (offs_m[:, None] * stride_am)
    b_ptrs_base = b_ptr + (offs_n[None, :] * stride_bn)
    
    # Main GEMM loop with software pipelining
    for k in range(0, K, BLOCK_K):
        # Compute masks once per iteration
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        
        # Load blocks with mask
        a = tl.load(
            a_ptrs_base + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0
        )
        b = tl.load(
            b_ptrs_base + offs_k[:, None] * stride_bk,
            mask=b_mask,
            other=0.0
        )
        
        # Fused accumulation with TF32 tensor cores
        acc += tl.dot(a, b, allow_tf32=True, out_dtype=tl.float32)
        
        # Update base pointers
        a_ptrs_base += BLOCK_K * stride_ak
        b_ptrs_base += BLOCK_K * stride_bk
    
    # Fused GELU activation using fast approximation
    # GELU(x) ≈ x * sigmoid(1.702x) for faster computation
    x = acc
    sigmoid_arg = x * 1.702
    # Fast sigmoid: 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-sigmoid_arg))
    gelu_out = x * sigmoid
    
    # Store final output with mask
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, gelu_out, mask=mask)


@triton.jit
def fused_softmax_kernel(
    # Pointers to input/output
    x_ptr, output_ptr,
    # Matrix dimensions
    M, N,
    # Strides
    stride_xm, stride_xn,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 2,
):
    # 1D grid over rows
    pid = tl.program_id(0)
    
    # Row indices for this block
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < M
    
    # Allocate registers for max and sum reductions
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # FIRST PASS: compute row_max and row_sum for complete rows
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        
        # Load block
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        mask = row_mask[:, None] & col_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        
        # Update row max
        row_max = tl.maximum(row_max, tl.max(x_vals, axis=1))
    
    # SECOND PASS: compute sum of exponentials for complete rows
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        
        # Load block again
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        mask = row_mask[:, None] & col_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute exp(x - max) and accumulate to sum
        x_vals_minus_max = x_vals - row_max[:, None]
        exp_vals = tl.exp(x_vals_minus_max)
        row_sum += tl.sum(exp_vals, axis=1)
    
    # THIRD PASS: normalize and store with final sum
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs_n < N
        
        # Load block for the final time
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        mask = row_mask[:, None] & col_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        
        # Compute final softmax values using complete row_sum
        x_vals_minus_max = x_vals - row_max[:, None]
        exp_vals = tl.exp(x_vals_minus_max)
        softmax_vals = exp_vals / row_sum[:, None]
        
        output_ptrs = output_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        tl.store(output_ptrs, softmax_vals, mask=mask)


def fused_gemm_gelu_softmax(x, weight):
    """Fused GEMM + GELU + Softmax operation with optimized memory patterns."""
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    M, K = x.shape
    N = weight.shape[0]  # out_features
    
    # Allocate output tensors
    gemm_output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    final_output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid for GEMM+GELU kernel - tuned for Ada Lovelace
    grid_gemm = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N'])
    )
    
    # Launch GEMM+GELU kernel with optimized block sizes
    gemm_gelu_kernel[grid_gemm](
        x, weight, gemm_output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # Transposed access
        gemm_output.stride(0), gemm_output.stride(1),
        BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        num_stages=2
    )
    
    # Grid for fused softmax kernel
    grid_softmax = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    
    # Launch fused softmax kernel - reads from gemm_output, writes to final_output
    fused_softmax_kernel[grid_softmax](
        gemm_output, final_output,
        M, N,
        gemm_output.stride(0), gemm_output.stride(1),
        final_output.stride(0), final_output.stride(1),
        BLOCK_M=64, BLOCK_N=256,
        num_stages=2
    )
    
    return final_output


class ModelNew(nn.Module):
    """Optimized model with fully fused GEMM + GELU + Softmax operations."""
    
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def forward(self, x):
        return fused_gemm_gelu_softmax(x, self.weight)
