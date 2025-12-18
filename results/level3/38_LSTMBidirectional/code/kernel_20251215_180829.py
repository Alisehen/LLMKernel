import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def fused_gemm_last_step_kernel(
    # Input pointers
    x_ptr,  # [B, T, K]
    weight_ptr,  # [N, K]
    bias_ptr,  # [N]
    # Output pointer
    out_ptr,  # [B, N]
    # Shapes
    B, T, K, N,
    # Strides
    stride_xb, stride_xt, stride_xk,
    stride_wk, stride_wn,
    stride_ob, stride_on,
    # Block constants
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel that:
    1. Selects the last timestep from x[:, -1, :] (shape [B, K])
    2. Performs matrix multiplication with weight [N, K]
    3. Adds bias [N]
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Output feature dimension
    
    # Offsets
    offs_b = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointer to last timestep (x[:, -1, :])
    # x_ptr + (T-1)*stride_xt gives the last timestep
    last_timestep_offset = (T - 1) * stride_xt
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Create masks
        b_mask = offs_b < B
        n_mask = offs_n < N
        k_mask = offs_k < (K - k)
        k_full_mask = offs_k < K
        
        # Load block of x [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_b[:, None] * stride_xb + last_timestep_offset + \
                (k + offs_k[None, :]) * stride_xk
        x_block = tl.load(x_ptrs, mask=b_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load block of weight [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + (k + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn
        w_block = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(x_block, w_block, allow_tf32=True)
    
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]
    
    # Store result
    out_ptrs = out_ptr + offs_b[:, None] * stride_ob + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=b_mask[:, None] & n_mask[None, :])


def fused_gemm_last_step(x, weight, bias):
    """
    Fused operation: x[:, -1, :] @ weight.t() + bias
    x: [B, T, K]
    weight: [N, K]
    bias: [N] or None
    """
    B, T, K = x.shape
    N = weight.shape[0]
    
    # Output tensor
    out = torch.empty((B, N), device=x.device, dtype=x.dtype)
    
    # Grid computation
    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    # Prepare weight for kernel (need K,N layout)
    weight_t = weight.t().contiguous()
    
    fused_gemm_last_step_kernel[grid](
        x, weight_t, bias, out,
        B, T, K, N,
        x.stride(0), x.stride(1), x.stride(2),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        # Keep LSTM as is (too complex to implement in Triton for now)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Replace nn.Linear with parameters for fused kernel
        self.weight = nn.Parameter(torch.randn(output_size, hidden_size * 2))
        self.bias = nn.Parameter(torch.randn(output_size))
    
    def forward(self, x, h0, c0):
        # LSTM forward (unchanged)
        out, hn = self.lstm(x, (h0, c0))
        
        # Fused: take last timestep and linear transformation
        return fused_gemm_last_step(out, self.weight, self.bias)
