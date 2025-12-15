import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8},
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8},
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4},
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_kernel(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    """
    Fused kernel for: out = relu(x @ w.T + bias)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tile from x (M x K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k * BLOCK_K + offs_k[None, :]) * stride_xk
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load tile from w (N x K) - note w is transposed in this kernel
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K - k * BLOCK_K)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k * BLOCK_K + offs_k[None, :]) * stride_wk
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=True)
    
    # Add bias
    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Apply ReLU
    acc = tl.where(acc > 0, acc, 0.0)
    
    # Write output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_linear_relu(x, weight, bias):
    """
    Wrapper function for fused linear + bias + ReLU operation.
    Computes: output = relu(x @ weight.T + bias)
    
    Args:
        x: Input tensor of shape (M, K)
        weight: Weight tensor of shape (N, K)
        bias: Bias tensor of shape (N,) or None
        
    Returns:
        Output tensor of shape (M, N)
    """
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Shape mismatch: x has K={K}, weight has K={K_w}"
    
    # Pre-transpose weight for better memory access pattern
    weight_t = weight.t().contiguous()
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Choose configuration based on data type
    if x.dtype == torch.float16:
        acc_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        acc_dtype = tl.bfloat16
    else:
        acc_dtype = tl.float32
    
    # Set grid and kernel launch parameters
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    # Launch kernel
    fused_linear_relu_kernel[grid](
        x_ptr=x,
        w_ptr=weight_t,
        b_ptr=bias if bias is not None else None,
        out_ptr=out,
        M=M,
        N=N,
        K=K,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_wk=weight_t.stride(0),
        stride_wn=weight_t.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        ACC_TYPE=acc_dtype,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Fused version of Model with optimized Triton kernel.
    Performs: out = relu(x @ weight.T + bias)
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return fused_linear_relu(x, self.weight, self.bias)
