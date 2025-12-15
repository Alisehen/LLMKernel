import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def fused_gemm_sigmoid_scale_resadd_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Scaling factor
    scaling_factor,
):
    # Program ID: which block of C we're computing
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for the first blocks of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute block of C = A * B
    for k in range(0, K, BLOCK_K):
        # Load blocks from A and B with masking
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k)
        b_mask = (offs_k[:, None] < K - k) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        accumulator += tl.dot(a, b, allow_tf32=True)
        
        # Move pointers to next blocks
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    accumulator += bias[None, :]
    
    # Store original output for residual connection
    original_output = accumulator
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Use stable computation for exp(-x): exp(min(-x, 0))
    neg_x = -accumulator
    exp_neg_x = tl.exp(tl.minimum(neg_x, 0.0))
    
    # Handle positive and negative cases separately for numerical stability
    # For x >= 0: sigmoid = 1 / (1 + exp(-x))
    # For x < 0: sigmoid = exp(x) / (1 + exp(x))
    pos_mask = accumulator >= 0.0
    neg_mask = ~pos_mask
    
    # Compute sigmoid for positive x
    sigmoid_pos = 1.0 / (1.0 + tl.where(pos_mask, exp_neg_x, 0.0))
    
    # Compute sigmoid for negative x
    exp_x = tl.exp(tl.minimum(accumulator, 0.0))  # exp(min(x, 0)) for stability
    sigmoid_neg = tl.where(neg_mask, exp_x / (1.0 + exp_x), 0.0)
    
    # Combine results
    sigmoid_result = tl.where(pos_mask, sigmoid_pos, sigmoid_neg)
    
    # Scale by scaling factor
    scaled_result = sigmoid_result * scaling_factor
    
    # Add residual: output = scaled_sigmoid + original_output
    output = scaled_result + original_output
    
    # Store output to C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, output, mask=c_mask)

def fused_gemm_sigmoid_scale_resadd(x, weight, bias, scaling_factor):
    """
    Fused operation: GEMM -> Sigmoid -> Scaling -> Residual Add
    
    Args:
        x: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Bias tensor of shape (N,)
        scaling_factor: Scaling factor for sigmoid output
        
    Returns:
        Output tensor of shape (M, N)
    """
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: x has K={K}, weight has K={K_w}"
    
    # Allocate output tensor
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Transpose weight for column-major access
    weight_t = weight.t().contiguous()
    
    # Grid dimensions
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
    
    # Launch kernel
    fused_gemm_sigmoid_scale_resadd_kernel[grid](
        x, weight_t, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        scaling_factor=scaling_factor,
    )
    
    return output

class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        return fused_gemm_sigmoid_scale_resadd(
            x, self.weight, self.bias, self.scaling_factor
        )
