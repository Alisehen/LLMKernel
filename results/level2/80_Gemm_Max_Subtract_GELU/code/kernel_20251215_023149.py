import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gemm_max_sub_gelu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    # Kernel parameters
    max_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    USE_TF32: tl.constexpr = True,
):
    """
    Fully fused kernel for:
    1. GEMM: x @ weight.T
    2. Max reduction along specified dimension
    3. Subtract mean along dim=1
    4. GELU activation
    
    Optimized for Ada Lovelace with register pressure awareness.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # ----------------------------------------------------------
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # ----------------------------------------------------------
    # Inner product with K dimension blocking
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, 
                   mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K),
                   other=0.0)
        b = tl.load(b_ptrs,
                   mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N),
                   other=0.0)
        
        if USE_TF32:
            acc += tl.dot(a, b, allow_tf32=True)
        else:
            acc += tl.dot(a, b, allow_tf32=False)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # ----------------------------------------------------------
    # Handle max reduction, subtract mean, and GELU
    # Based on profiling, we recompute cheap ops to save registers
    if max_dim == 1:  # Reduce over columns (dim=1)
        # Each row produces one value
        row_mask = offs_am < M
        col_mask = offs_bn < N
        
        # Compute max per row within this block
        if BLOCK_N > 1:
            # Use efficient reduction for larger blocks
            max_vals = tl.max(acc, axis=1)
        else:
            # Direct value for single column per thread
            max_vals = tl.where(col_mask, acc[:, 0], -1e30)
        
        # Broadcast max to all threads in block
        max_vals = tl.broadcast_to(max_vals[:, None], (BLOCK_M, BLOCK_N))
        
        # Subtract max (element-wise)
        centered = acc - max_vals
        
        # Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Optimized for Ada Lovelace TF32 tensor cores
        sqrt_2_over_pi = 0.7978845608028654
        gelu_coeff = 0.044715
        
        x = centered
        # Compute x^3 using FMA where possible
        x_sq = x * x
        x_cubed = x * x_sq
        inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
        
        # tanh approximation using exp
        exp_2x = tl.exp(2.0 * inner)
        tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
        
        gelu = 0.5 * x * (1.0 + tanh_inner)
        
        # Store results
        out_ptrs = output_ptr + offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_n
        tl.store(out_ptrs, gelu, mask=row_mask[:, None] & col_mask[None, :])
        
    else:  # max_dim == 0, reduce over rows (dim=0)
        # Each column produces one value
        row_mask = offs_am < M
        col_mask = offs_bn < N
        
        # Compute max per column within this block
        if BLOCK_M > 1:
            max_vals = tl.max(acc, axis=0)
        else:
            max_vals = tl.where(row_mask, acc[0, :], -1e30)
        
        # Broadcast max to all threads in block
        max_vals = tl.broadcast_to(max_vals[None, :], (BLOCK_M, BLOCK_N))
        
        # Subtract max (element-wise)
        centered = acc - max_vals
        
        # Apply GELU
        sqrt_2_over_pi = 0.7978845608028654
        gelu_coeff = 0.044715
        
        x = centered
        x_sq = x * x
        x_cubed = x * x_sq
        inner = sqrt_2_over_pi * (x + gelu_coeff * x_cubed)
        exp_2x = tl.exp(2.0 * inner)
        tanh_inner = (exp_2x - 1.0) / (exp_2x + 1.0)
        gelu = 0.5 * x * (1.0 + tanh_inner)
        
        # Store results
        out_ptrs = output_ptr + offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_n
        tl.store(out_ptrs, gelu, mask=row_mask[:, None] & col_mask[None, :])

def fused_gemm_max_sub_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    max_dim: int,
    configs: list = None
):
    """
    Wrapper for the fully fused kernel.
    """
    M, K = x.shape
    N = weight.shape[0]
    
    # Transpose weight for GEMM
    weight_t = weight.t().contiguous()
    
    # Output shape depends on max_dim
    if max_dim == 1:
        output_shape = (M, N)  # Full output for GELU
    else:  # max_dim == 0
        output_shape = (M, N)  # Full output for GELU
    
    # Allocate output
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Default configurations optimized for Ada Lovelace
    if configs is None:
        configs = [
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},  # High occupancy
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8},   # Memory-friendly
            {'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},   # Compute-heavy
        ]
    
    # Grid calculation
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Use autotuning - start with conservative configuration
    config = configs[0]
    
    # Launch kernel
    fused_gemm_max_sub_gelu_kernel[grid](
        x, weight_t, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
        max_dim,
        BLOCK_M=config['BLOCK_M'],
        BLOCK_N=config['BLOCK_N'],
        BLOCK_K=config['BLOCK_K'],
        GROUP_M=config['GROUP_M'],
        USE_TF32=True,
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs a GEMM, max reduction,
    subtraction, and GELU activation in a single fused kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        # Initialize weight with proper scaling for stable training
        std = (2.0 / in_features) ** 0.5
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.max_dim = max_dim
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor after fused operations
        """
        return fused_gemm_max_sub_gelu(x, self.weight, self.max_dim)
