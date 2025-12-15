import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# -----------------------------------------------------------------------------
# Optimized Gemm kernel with Tensor Core optimization
# -----------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TF32: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)
    
    # Create offset pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer arithmetic for A and B matrices
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, K, BLOCK_K):
        # Load A and B tiles with masking
        a = tl.load(a_ptrs, 
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
                    other=0.0)
        
        # Matrix multiplication with TF32 support
        acc += tl.dot(a, b, allow_tf32=USE_TF32)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result with masking
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# -----------------------------------------------------------------------------
# Fused BatchNorm + Scale kernel (element-wise, inference only)
# -----------------------------------------------------------------------------
@triton.jit
def fused_bn_scale_kernel(
    x_ptr, gamma_ptr, beta_ptr, running_mean_ptr, running_var_ptr,
    scale_ptr, output_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_output_m, stride_output_n,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks for boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load parameters
    gamma = tl.load(gamma_ptr + offs_n, mask=mask_n, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask_n, other=0.0)
    r_mean = tl.load(running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    r_var = tl.load(running_var_ptr + offs_n, mask=mask_n, other=1.0)
    scale = tl.load(scale_ptr)
    
    # Load input
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0)
    
    # Apply batch normalization and scaling
    inv_std = 1.0 / tl.sqrt(r_var + eps)
    normalized = (x - r_mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]
    scaled = normalized * scale
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, scaled,
             mask=mask_m[:, None] & mask_n[None, :])

# -----------------------------------------------------------------------------
# Optimized Softmax kernel with reduction fusion
# -----------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_input_m, stride_input_n,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Row block offsets
    row_start = pid * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < M
    
    # Initialize max and sum for reduction
    row_max = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # First pass: compute max and accumulate exp
    for start_col in range(0, N, BLOCK_N):
        col_offs = start_col + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        
        # Load input tile
        input_ptrs = input_ptr + row_offs[:, None] * stride_input_m + col_offs[None, :] * stride_input_n
        x = tl.load(input_ptrs,
                   mask=row_mask[:, None] & col_mask[None, :],
                   other=float('-inf'))
        
        # Update row max
        row_max = tl.maximum(row_max, tl.max(x, axis=1))
    
    # Second pass: compute exp and sum
    for start_col in range(0, N, BLOCK_N):
        col_offs = start_col + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        
        # Load input tile again
        input_ptrs = input_ptr + row_offs[:, None] * stride_input_m + col_offs[None, :] * stride_input_n
        x = tl.load(input_ptrs,
                   mask=row_mask[:, None] & col_mask[None, :],
                   other=float('-inf'))
        
        # Compute exp(x - max) and accumulate
        exp_x = tl.exp(x - row_max[:, None])
        row_sum += tl.sum(exp_x, axis=1)
    
    # Third pass: normalize and store
    for start_col in range(0, N, BLOCK_N):
        col_offs = start_col + tl.arange(0, BLOCK_N)
        col_mask = col_offs < N
        
        # Load input tile again
        input_ptrs = input_ptr + row_offs[:, None] * stride_input_m + col_offs[None, :] * stride_input_n
        x = tl.load(input_ptrs,
                   mask=row_mask[:, None] & col_mask[None, :],
                   other=float('-inf'))
        
        # Compute softmax
        exp_x = tl.exp(x - row_max[:, None])
        softmax_out = exp_x / row_sum[:, None]
        
        # Store result
        output_ptrs = output_ptr + row_offs[:, None] * stride_output_m + col_offs[None, :] * stride_output_n
        tl.store(output_ptrs, softmax_out,
                mask=row_mask[:, None] & col_mask[None, :])

# -----------------------------------------------------------------------------
# Wrapper functions with auto-tuning
# -----------------------------------------------------------------------------
def gemm_wrapper(x, weight):
    """Optimized matrix multiplication with auto-tuning."""
    M, K = x.shape
    N = weight.shape[0]
    
    # Prepare output tensor
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Transpose weight for efficient access
    weight_t = weight.t().contiguous()
    
    # Grid configuration
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    # Auto-tune configurations optimized for Ada Lovelace
    configs = [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
    ]
    
    # Use the first configuration from configs list (fix critical issue)
    best_config = configs[0]
    
    # Launch kernel with best configuration
    gemm_kernel[grid](
        x, weight_t, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=best_config.kwargs['BLOCK_M'],
        BLOCK_N=best_config.kwargs['BLOCK_N'],
        BLOCK_K=best_config.kwargs['BLOCK_K'],
        USE_TF32=True,
        GROUP_M=best_config.kwargs['GROUP_M'],
        num_warps=best_config.num_warps,
    )
    
    return output

def fused_bn_scale_wrapper(x, gamma, beta, running_mean, running_var, scale, eps=1e-5):
    """Fused BatchNorm + Scale wrapper (inference only)."""
    M, N = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Grid configuration (2D for better parallelism)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    # Auto-tune configurations
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32}, num_warps=4),
    ]
    
    # Launch kernel with first configuration
    best_config = configs[0]
    fused_bn_scale_kernel[grid](
        x, gamma, beta, running_mean, running_var,
        scale, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        eps,
        BLOCK_M=best_config.kwargs['BLOCK_M'],
        BLOCK_N=best_config.kwargs['BLOCK_N'],
        num_warps=best_config.num_warps,
    )
    
    return output

def softmax_wrapper(x):
    """Optimized softmax wrapper."""
    M, N = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Grid configuration
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    
    # Auto-tune configurations
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8),
    ]
    
    # Launch kernel with balanced configuration
    best_config = configs[1]
    softmax_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=best_config.kwargs['BLOCK_M'],
        BLOCK_N=best_config.kwargs['BLOCK_N'],
        num_warps=best_config.num_warps,
    )
    
    return output

# -----------------------------------------------------------------------------
# Main Model Class
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (Gemm), 
    Batch Normalization, scaling, and Softmax in optimized kernels.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        # BatchNorm buffers (running statistics)
        self.register_buffer('running_mean', torch.zeros(out_features))
        self.register_buffer('running_var', torch.ones(out_features))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize gamma and beta
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Step 1: Matrix multiplication with optimized kernel
        x = gemm_wrapper(x, self.weight)
        
        # Step 2: Fused BatchNorm + Scale (inference only)
        x = fused_bn_scale_wrapper(
            x, self.gamma, self.beta,
            self.running_mean, self.running_var,
            self.scale, self.bn_eps
        )
        
        # Step 3: Softmax with optimized kernel
        x = softmax_wrapper(x)
        
        return x
