import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Mish activation: x * tanh(softplus(x))
    softplus = tl.log(1.0 + tl.exp(x))
    exp_2s = tl.exp(2.0 * softplus)
    tanh_softplus = (exp_2s - 1.0) / (exp_2s + 1.0)
    mish_out = x * tanh_softplus
    
    tl.store(output_ptr + offsets, mish_out, mask=mask)

@triton.jit
def double_mish_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_base = pid * BLOCK_SIZE * VEC_SIZE
    
    # Vectorized processing
    offsets = offs_base + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    offsets = tl.view(offsets, BLOCK_SIZE * VEC_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # First Mish
    softplus1 = tl.log(1.0 + tl.exp(x))
    exp_2s1 = tl.exp(2.0 * softplus1)
    tanh_softplus1 = (exp_2s1 - 1.0) / (exp_2s1 + 1.0)
    y = x * tanh_softplus1
    
    # Second Mish
    softplus2 = tl.log(1.0 + tl.exp(y))
    exp_2s2 = tl.exp(2.0 * softplus2)
    tanh_softplus2 = (exp_2s2 - 1.0) / (exp_2s2 + 1.0)
    z = y * tanh_softplus2
    
    tl.store(output_ptr + offsets, z, mask=mask)

@triton.jit
def double_mish_kernel_2d(
    input_ptr,
    output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load input
    input_ptrs = input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # First Mish
    softplus1 = tl.log(1.0 + tl.exp(x))
    exp_2s1 = tl.exp(2.0 * softplus1)
    tanh_softplus1 = (exp_2s1 - 1.0) / (exp_2s1 + 1.0)
    y = x * tanh_softplus1
    
    # Second Mish
    softplus2 = tl.log(1.0 + tl.exp(y))
    exp_2s2 = tl.exp(2.0 * softplus2)
    tanh_softplus2 = (exp_2s2 - 1.0) / (exp_2s2 + 1.0)
    z = y * tanh_softplus2
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    tl.store(output_ptrs, z, mask=mask)

def triton_mish(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Autotune configuration
    configs = [
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ]
    
    @triton.autotune(configs=configs, key=['n_elements'])
    @triton.jit
    def kernel_wrapper(
        input_ptr, output_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr
    ):
        mish_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE)
    
    kernel_wrapper[grid](x, output, n_elements)
    return output

def triton_double_mish(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Choose kernel based on tensor shape
    if x.dim() >= 2:
        M, N = x.shape[-2], x.shape[-1]
        total_rows = 1
        for dim in x.shape[:-2]:
            total_rows *= dim
        
        def grid_2d(meta):
            return (total_rows, triton.cdiv(N, meta['BLOCK_N']))
        
        # 2D kernel with autotune
        configs = [
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 128}, num_warps=4, num_stages=1),
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256}, num_warps=4, num_stages=1),
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512}, num_warps=8, num_stages=1),
            triton.Config({'BLOCK_M': 1, 'BLOCK_N': 1024}, num_warps=8, num_stages=1),
        ]
        
        @triton.autotune(configs=configs, key=['M', 'N'])
        @triton.jit
        def kernel_2d_wrapper(
            input_ptr, output_ptr, M, N,
            stride_m, stride_n,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr
        ):
            double_mish_kernel_2d(input_ptr, output_ptr, M, N, stride_m, stride_n, BLOCK_M, BLOCK_N)
        
        # Reshape to 3D for batch processing
        x_2d = x.view(total_rows, M, N)
        output_2d = output.view(total_rows, M, N)
        
        for i in range(total_rows):
            kernel_2d_wrapper[grid_2d](
                x_2d[i], output_2d[i], M, N,
                x_2d.stride(1), x_2d.stride(2)
            )
    else:
        # 1D vectorized kernel
        def grid_1d(meta):
            return (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VEC_SIZE']),)
        
        # Autotune configuration for 1D
        configs = [
            triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 4}, num_warps=4, num_stages=1),
            triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 4}, num_warps=8, num_stages=1),
            triton.Config({'BLOCK_SIZE': 1024, 'VEC_SIZE': 4}, num_warps=8, num_stages=1),
            triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 8}, num_warps=4, num_stages=1),
            triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 8}, num_warps=8, num_stages=1),
        ]
        
        @triton.autotune(configs=configs, key=['n_elements'])
        @triton.jit
        def kernel_1d_wrapper(
            input_ptr, output_ptr, n_elements,
            BLOCK_SIZE: tl.constexpr,
            VEC_SIZE: tl.constexpr
        ):
            double_mish_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE, VEC_SIZE)
        
        kernel_1d_wrapper[grid_1d](x, output, n_elements)
    
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    Uses fused Triton kernels for the Mish activations.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Use optimized double mish kernel
        x = triton_double_mish(x)
        return x
