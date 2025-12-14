import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_N': 64},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=['B', 'K', 'N'],
)
@triton.jit
def fused_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    subtract_ptr,
    output_ptr,
    B,
    N,
    K,
    stride_input_b,
    stride_input_n,
    stride_weight_k,
    stride_weight_n,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    # Block offsets
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Masks
    mask_b = offs_b < B
    mask_k = offs_k < K
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over N dimension for matrix multiplication
    for n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        
        # Load input block [BLOCK_SIZE_B, BLOCK_SIZE_N]
        input_ptrs = (
            input_ptr + 
            offs_b[:, None] * stride_input_b + 
            offs_n[None, :] * stride_input_n
        )
        input_block = tl.load(
            input_ptrs, 
            mask=mask_b[:, None] & mask_n[None, :],
            other=0.0
        )
        
        # Load weight block [BLOCK_SIZE_N, BLOCK_SIZE_K]
        weight_ptrs = (
            weight_ptr + 
            offs_n[:, None] * stride_weight_n + 
            offs_k[None, :] * stride_weight_k
        )
        weight_block = tl.load(
            weight_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Accumulate matrix multiplication
        accumulator += tl.dot(input_block, weight_block, allow_tf32=False)
    
    # Add bias if present
    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_k[None, :]
        bias = tl.load(bias_ptrs, mask=mask_k[None, :], other=0.0)
        accumulator += bias
    
    # Subtract parameter
    subtract_ptrs = subtract_ptr + offs_k[None, :]
    subtract = tl.load(subtract_ptrs, mask=mask_k[None, :], other=0.0)
    accumulator -= subtract
    
    # Store result [B, K]
    output_ptrs = (
        output_ptr + 
        offs_b[:, None] * K + 
        offs_k[None, :]
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=mask_b[:, None] & mask_k[None, :]
    )


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_K': 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_K': 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_K': 128},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=['B', 'K'],
)
@triton.jit
def reduce_and_activate_kernel(
    gemm_output_ptr,
    original_input_ptr,
    final_output_ptr,
    B,
    K,
    stride_gemm_b,
    stride_gemm_k,
    stride_input_b,
    stride_input_k,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = offs_b < B
    
    # Accumulate sum for mean calculation
    block_sum = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    block_count = 0
    
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load gemm output values
        gemm_ptrs = (
            gemm_output_ptr + 
            offs_b[:, None] * stride_gemm_b + 
            offs_k[None, :] * stride_gemm_k
        )
        vals = tl.load(
            gemm_ptrs,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Accumulate sum
        block_sum += tl.sum(vals, axis=1)
        block_count += tl.sum(mask_k, axis=0)
    
    # Calculate mean (GlobalAvgPool)
    mean_val = block_sum / K
    
    # LogSumExp on single value dimension
    # Since we have [B,1], logsumexp is just the value itself
    lse_val = mean_val
    
    # GELU approximation
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coef = 0.044715
    
    x = lse_val
    x_sq = x * x
    x_cubed = x * x_sq
    inner = sqrt_2_over_pi * (x + gelu_coef * x_cubed)
    # tanh approximation
    tanh_inner = tl.where(
        inner > 0,
        1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0),
        -1.0 + 2.0 / (tl.exp(-2.0 * inner) + 1.0)
    )
    gelu_val = 0.5 * x * (1.0 + tanh_inner)
    
    # Broadcast gelu_val and add to original input
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # Load original input
        input_ptrs = (
            original_input_ptr + 
            offs_b[:, None] * stride_input_b + 
            offs_k[None, :] * stride_input_k
        )
        orig_vals = tl.load(
            input_ptrs,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Add broadcasted gelu value
        result = orig_vals + gelu_val[:, None]
        
        # Store final result
        output_ptrs = (
            final_output_ptr + 
            offs_b[:, None] * K + 
            offs_k[None, :]
        )
        tl.store(
            output_ptrs,
            result,
            mask=mask_b[:, None] & mask_k[None, :]
        )


def triton_fused_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    B, N = x.shape
    K = weight.shape[0]
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    subtract = subtract.contiguous()
    
    # Allocate intermediate output
    gemm_output = torch.empty((B, K), device=x.device, dtype=x.dtype)
    
    # Launch first kernel
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_SIZE_B']),
        triton.cdiv(K, meta['BLOCK_SIZE_K']),
    )
    
    fused_forward_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,  # Dummy pointer if no bias
        subtract,
        gemm_output,
        B,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        USE_BIAS=bias is not None,
    )
    
    # Allocate final output
    final_output = torch.empty_like(x)
    
    # Launch second kernel
    grid_red = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']),)
    
    reduce_and_activate_kernel[grid_red](
        gemm_output,
        x,
        final_output,
        B,
        K,
        gemm_output.stride(0),
        gemm_output.stride(1),
        x.stride(0),
        x.stride(1),
    )
    
    return final_output


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        
        # Store references for kernel
        self._weight = self.gemm.weight
        self._bias = self.gemm.bias

    def forward(self, x):
        original_x = x
        
        x = triton_fused_forward(
            x,
            self._weight,
            self._bias,
            self.subtract,
        )
        
        return x
