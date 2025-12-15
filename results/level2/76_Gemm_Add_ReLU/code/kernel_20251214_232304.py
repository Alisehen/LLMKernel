import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small block config for high occupancy, low register pressure
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Medium block config - balanced for most cases
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # Larger configs for big matrices
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets with proper masking
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to the first blocks of X and W
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles with masking
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K - k * BLOCK_K)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Matrix multiplication with TF32 Tensor Cores
        acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=True)
        
        # Move pointers to next K block
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Add bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # ReLU activation - computed in-place to save registers
    acc = tl.maximum(acc, 0.0)
    
    # Store output with masking
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_linear_relu(x, weight, bias):
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w
    
    # Transpose weight for column-major access
    weight_t = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Use float32 accumulation for better precision and Tensor Core utilization
    if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
        acc_dtype = tl.float32
    else:
        acc_dtype = tl.float32
    
    # Grid function
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
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
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        return fused_linear_relu(x, self.weight, self.bias)
