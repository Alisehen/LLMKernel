import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_forward_kernel(
    # Pointers to input and parameters
    input_ptr,
    weight_ptr,
    bias_ptr,
    subtract_ptr,
    output_ptr,
    # Matrix dimensions
    B,  # batch size
    N,  # input features
    K,  # output features (must equal N for residual add)
    # Strides
    stride_input_batch,
    stride_input_feat,
    stride_weight_in,
    stride_weight_out,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_TMA: tl.constexpr,
):
    # Parallelize over batches and output features with 2D grid
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # Block indices
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for boundary checks
    mask_b = offs_b < B
    mask_k = offs_k < K
    
    # Initialize accumulator for GEMM
    accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Use tensor cores for FP16/BF16
    if USE_TMA:
        # Load input and weight blocks with tensor core friendly layout
        for n_block in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            offs_n = n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_n = offs_n < N
            
            # Load input block
            input_ptrs = (
                input_ptr + 
                offs_b[:, None] * stride_input_batch + 
                offs_n[None, :] * stride_input_feat
            )
            input_block = tl.load(
                input_ptrs, 
                mask=mask_b[:, None] & mask_n[None, :],
                other=0.0
            ).to(tl.float16)
            
            # Load weight block
            weight_ptrs = (
                weight_ptr + 
                offs_n[:, None] * stride_weight_in + 
                offs_k[None, :] * stride_weight_out
            )
            weight_block = tl.load(
                weight_ptrs,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0
            ).to(tl.float16)
            
            # Tensor core accelerated dot product
            accumulator += tl.dot(input_block, weight_block, out_dtype=tl.float32)
    else:
        # Standard implementation for fallback
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
    
    # Apply bias if needed
    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_k[None, :]
        bias = tl.load(bias_ptrs, mask=mask_k[None, :], other=0.0)
        accumulator += bias
    
    # Apply subtract parameter (broadcast over batch dimension)
    subtract_ptrs = subtract_ptr + offs_k[None, :]
    subtract = tl.load(subtract_ptrs, mask=mask_k[None, :], other=0.0)
    accumulator -= subtract
    
    # Store to temporary output for reductions
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


@triton.jit
def reduce_and_activate_kernel(
    # Pointers
    gemm_output_ptr,
    original_input_ptr,
    final_output_ptr,
    # Dimensions
    B,
    K,
    # Strides
    stride_gemm_batch,
    stride_gemm_feat,
    stride_input_batch,
    stride_input_feat,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Parallelize over batches with 2D grid
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    mask_b = offs_b < B
    mask_k = offs_k < K
    
    # Load gemm output values for this block
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
    
    # Partial sum for this block
    partial_sum = tl.sum(vals, axis=1)
    
    # Use shared memory for reduction across K blocks
    shmem = tl.static_shared_memory((BLOCK_SIZE_B,), tl.float32)
    
    # Initialize shared memory
    if tl.program_id(1) == 0:
        shmem[offs_b] = 0.0
    tl.debug_barrier()
    
    # Atomic add to shared memory
    tl.atomic_add(shmem + offs_b, partial_sum, mask=mask_b)
    tl.debug_barrier()
    
    # Only the first K-block computes final result
    if pid_k == 0:
        # Load total sum from shared memory
        total_sum = tl.load(shmem + offs_b, mask=mask_b)
        
        # Compute mean (divide by K)
        mean_val = total_sum / K
        
        # Fast GELU approximation optimized for Triton
        x = mean_val
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x_sq = x * x
        x_cubed = x * x_sq
        inner = 0.7978845608028654 * (x + 0.044715 * x_cubed)
        tanh_inner = tl.where(
            inner > 0,
            1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0),
            2.0 / (tl.exp(-2.0 * inner) + 1.0) - 1.0
        )
        gelu_val = 0.5 * x * (1.0 + tanh_inner)
        
        # Residual add: add gelu_val[batch] to each element of original input row
        # Process multiple K blocks per thread block for better efficiency
        for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K * tl.num_programs(1))):
            k_start = k_idx * BLOCK_SIZE_K * tl.num_programs(1) + tl.program_id(1) * BLOCK_SIZE_K
            offs_k_loop = k_start + tl.arange(0, BLOCK_SIZE_K)
            mask_k_loop = offs_k_loop < K
            
            # Load original input
            input_ptrs = (
                original_input_ptr + 
                offs_b[:, None] * stride_input_batch + 
                offs_k_loop[None, :] * stride_input_feat
            )
            orig_vals = tl.load(
                input_ptrs,
                mask=mask_b[:, None] & mask_k_loop[None, :],
                other=0.0
            )
            
            # Add GELU value (broadcast over features)
            result = orig_vals + gelu_val[:, None]
            
            # Store final output
            output_ptrs = (
                final_output_ptr + 
                offs_b[:, None] * K + 
                offs_k_loop[None, :]
            )
            tl.store(
                output_ptrs,
                result,
                mask=mask_b[:, None] & mask_k_loop[None, :]
            )


def triton_fused_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract: torch.Tensor,
) -> torch.Tensor:
    """
    Fused forward pass implementing:
    1. GEMM (linear layer)
    2. Subtract parameter
    3. Global average pool (mean over features)
    4. LogSumExp (simplified to identity after mean)
    5. GELU activation
    6. Residual add with original input
    """
    B, N = x.shape
    K = weight.shape[0]
    
    # Ensure K == N for residual add
    if K != N:
        raise ValueError(f"Output features ({K}) must equal input features ({N}) for residual add")
    
    # Check if tensor cores can be used
    use_tma = x.dtype in [torch.float16, torch.bfloat16] and weight.dtype in [torch.float16, torch.bfloat16]
    
    # Intermediate storage for GEMM output
    gemm_output = torch.empty((B, K), device=x.device, dtype=x.dtype)
    
    # Autotune configurations for GEMM kernel
    configs = [
        {'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'num_warps': 8, 'num_stages': 3},
        {'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'num_warps': 8, 'num_stages': 4},
        {'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'num_warps': 8, 'num_stages': 3},
    ]
    
    @triton.autotune(
        configs=configs,
        key=['B', 'N', 'K'],
    )
    @triton.jit
    def tuned_fused_forward_kernel(
        input_ptr, weight_ptr, bias_ptr, subtract_ptr, output_ptr,
        B, N, K,
        stride_input_batch, stride_input_feat, stride_weight_in, stride_weight_out,
        BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        USE_BIAS: tl.constexpr, USE_TMA: tl.constexpr,
        num_warps: tl.constexpr, num_stages: tl.constexpr
    ):
        # Reuse the main kernel logic
        fused_forward_kernel(
            input_ptr, weight_ptr, bias_ptr, subtract_ptr, output_ptr,
            B, N, K,
            stride_input_batch, stride_input_feat, stride_weight_in, stride_weight_out,
            BLOCK_SIZE_B, BLOCK_SIZE_K, BLOCK_SIZE_N, USE_BIAS, USE_TMA
        )
    
    # Launch GEMM kernel with autotuning
    grid_gemm = lambda META: (
        triton.cdiv(B, META['BLOCK_SIZE_B']),
        triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    
    tuned_fused_forward_kernel[grid_gemm](
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
        USE_TMA=use_tma,
    )
    
    # Allocate final output
    final_output = torch.empty_like(x)
    
    # Configure reduction and activation kernel
    BLOCK_SIZE_B_RED = 128
    BLOCK_SIZE_K_RED = 64
    
    # Use 2D grid for better parallelism
    grid_red = (
        triton.cdiv(B, BLOCK_SIZE_B_RED),
        min(4, triton.cdiv(K, BLOCK_SIZE_K_RED)),
    )
    
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
        BLOCK_SIZE_B=BLOCK_SIZE_B_RED,
        BLOCK_SIZE_K=BLOCK_SIZE_K_RED,
        num_warps=8,
        num_stages=3,
    )
    
    return final_output


class ModelNew(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    Optimized with fused Triton kernels.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        
        # Store original weight and bias as buffers for Triton kernel
        # (not as parameters to avoid dtype conflicts)
        self.register_buffer('_weight', self.gemm.weight.data.half())
        if bias:
            self.register_buffer('_bias', self.gemm.bias.data.half())
        else:
            self._bias = None

    def forward(self, x):
        original_x = x
        
        # Convert to half precision for tensor cores if needed
        x_half = x.half() if x.dtype != torch.float16 else x
        
        # Use stored half-precision buffers
        subtract_half = self.subtract.half() if self.subtract.dtype != torch.float16 else self.subtract
        
        # Fused forward pass with Triton kernels
        x = triton_fused_forward(
            x_half,
            self._weight,
            self._bias,
            subtract_half,
        )
        
        # Convert back to original precision if needed
        if original_x.dtype != torch.float16:
            x = x.to(original_x.dtype)
        
        return x
