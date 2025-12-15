import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Optimized for Ada Lovelace Tensor Cores (8.9 compute capability)
        # Start with num_stages=2 as recommended, only increase if needed
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=4),
        # Larger blocks for better Tensor Core utilization
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=2, num_warps=8),
        # For very large matrices, try tiling in K dimension
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 2}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_relu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block dimensions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    # Meta-parameters
    USE_BIAS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # `pid` is split along M, N, and K for split-K decomposition
    # -----------------------------------------------------------
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B
    # that we will move through
    # ----------------------------------------------------------
    # Offsets for the block
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers to A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # ----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # Accumulate into `acc` with proper dtype
    # ----------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    
    # Loop over K with vectorization for better memory efficiency
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load A and B tiles with masking
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K)
        b_mask = (offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate with proper Tensor Core usage
        # Using allow_tf32=True for Ada Lovelace TF32 tensor cores
        # and out_dtype=ACC_TYPE for proper accumulation
        acc += tl.dot(a, b, allow_tf32=True, out_dtype=ACC_TYPE)
        
        # Advance pointers to next K block
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # ----------------------------------------------------------
    # Add bias if needed (fused operation - no intermediate store)
    # ----------------------------------------------------------
    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        acc += bias[None, :]
    
    # ----------------------------------------------------------
    # Apply ReLU activation (fused operation - no intermediate store)
    # ----------------------------------------------------------
    acc = tl.maximum(acc, 0.0)
    
    # ----------------------------------------------------------
    # Write back the block of the output matrix C
    # ----------------------------------------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    # Check dimensions
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Incompatible dimensions: {K} != {K_w}"
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Determine accumulation dtype for optimal Tensor Core usage
    if x.dtype in [torch.float16, torch.bfloat16]:
        acc_dtype = tl.float32  # Use FP32 accumulation for FP16/BF16 inputs
    else:
        acc_dtype = tl.float32  # Default to FP32 accumulation
    
    # Grid: 1D launch with M*N blocks for better load balancing
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Transpose weight for optimal memory access pattern
    weight_t = weight.t().contiguous()
    
    # Launch kernel
    fused_linear_relu_kernel[grid](
        a_ptr=x,
        b_ptr=weight_t,
        c_ptr=output,
        bias_ptr=bias if bias is not None else None,
        M=M,
        N=N,
        K=K,
        stride_am=x.stride(0),
        stride_ak=x.stride(1),
        stride_bk=weight_t.stride(0),
        stride_bn=weight_t.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        USE_BIAS=bias is not None,
        ACC_TYPE=acc_dtype,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        return fused_linear_relu(x, self.weight, self.bias)
