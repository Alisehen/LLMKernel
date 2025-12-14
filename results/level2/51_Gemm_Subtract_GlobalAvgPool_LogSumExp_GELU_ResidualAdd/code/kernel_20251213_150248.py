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
    stride_input_batch,
    stride_input_feat,
    stride_weight_in,
    stride_weight_out,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    mask_b = offs_b < B
    mask_k = offs_k < K
    
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        
        input_ptrs = (
            input_ptr + 
            offs_b[:, None] * stride_input_batch + 
            offs_n[None, :] * stride_input_feat
        )
        input_block = tl.load(
            input_ptrs, 
            mask=mask_b[:, None] & mask_n[None, :],
            other=0.0
        )
        
        weight_ptrs = (
            weight_ptr + 
            offs_n[:, None] * stride_weight_in + 
            offs_k[None, :] * stride_weight_out
        )
        weight_block = tl.load(
            weight_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        
        accumulator += tl.dot(input_block, weight_block)
    
    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_k[None, :]
        bias = tl.load(bias_ptrs, mask=mask_k[None, :], other=0.0)
        accumulator += bias
    
    subtract_ptrs = subtract_ptr + offs_k[None, :]
    subtract = tl.load(subtract_ptrs, mask=mask_k[None, :], other=0.0)
    accumulator -= subtract
    
    gemm_output_ptrs = (
        output_ptr + 
        offs_b[:, None] * K + 
        offs_k[None, :]
    )
    tl.store(
        gemm_output_ptrs,
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
    stride_gemm_batch,
    stride_gemm_feat,
    stride_input_batch,
    stride_input_feat,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = offs_b < B
    
    block_vals = tl.zeros((BLOCK_SIZE_B,), dtype=tl.float32)
    
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        gemm_ptrs = (
            gemm_output_ptr + 
            offs_b[:, None] * stride_gemm_batch + 
            offs_k[None, :] * stride_gemm_feat
        )
        vals = tl.load(
            gemm_ptrs,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0
        )
        
        block_vals += tl.sum(vals, axis=1)
    
    mean_val = block_vals / K
    
    sqrt_2_over_pi = 0.7978845608028654
    gelu_coef = 0.044715
    
    x = mean_val
    x_sq = x * x
    x_cubed = x * x_sq
    inner = sqrt_2_over_pi * (x + gelu_coef * x_cubed)
    exp_2inner = tl.exp(2.0 * inner)
    tanh_inner = 1.0 - 2.0 / (exp_2inner + 1.0)
    gelu_val = 0.5 * x * (1.0 + tanh_inner)
    
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        input_ptrs = (
            original_input_ptr + 
            offs_b[:, None] * stride_input_batch + 
            offs_k[None, :] * stride_input_feat
        )
        orig_vals = tl.load(
            input_ptrs,
            mask=mask_b[:, None] & mask_k[None, :],
            other=0.0
        )
        
        result = orig_vals + gelu_val[:, None]
        
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
    
    if K != N:
        raise ValueError(f"Output features ({K}) must equal input features ({N}) for residual add")
    
    gemm_output = torch.empty((B, K), device=x.device, dtype=x.dtype)
    
    grid = (
        triton.cdiv(B, 64),  # Initial estimate, actual size determined by autotune
        triton.cdiv(K, 64),
    )
    
    fused_forward_kernel[grid](
        x,
        weight,
        bias,
        subtract,
        gemm_output,
        B,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),
        weight.stride(0),
        USE_BIAS=bias is not None,
    )
    
    final_output = torch.empty_like(x)
    
    grid_red = (triton.cdiv(B, 64),)  # Initial estimate
    
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
        
        self._weight = self.gemm.weight
        self._bias = self.gemm.bias if bias else None

    def forward(self, x):
        original_x = x
        
        x = triton_fused_forward(
            x,
            self._weight,
            self._bias,
            self.subtract,
        )
        
        return x
