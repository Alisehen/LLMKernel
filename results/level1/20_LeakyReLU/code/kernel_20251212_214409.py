import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'VEC_WIDTH': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_WIDTH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'VEC_WIDTH': 2}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_WIDTH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 2}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def leaky_relu_kernel_optimized_vec(
    x_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_WIDTH: tl.constexpr,
):
    """Optimized LeakyReLU kernel with vectorized memory access."""
    pid = tl.program_id(axis=0)
    
    # Vectorized block processing - each thread handles VEC_WIDTH elements
    block_start = pid * BLOCK_SIZE * VEC_WIDTH
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_WIDTH + tl.arange(0, VEC_WIDTH)[None, :]
    
    # Flatten for vectorized load/store
    offsets_flat = tl.reshape(offsets, (BLOCK_SIZE * VEC_WIDTH,))
    mask_flat = offsets_flat < n_elements
    
    # Vectorized load with proper masking
    x_vals = tl.load(x_ptr + offsets_flat, mask=mask_flat, other=0.0)
    
    # Branchless LeakyReLU
    output_vals = tl.where(x_vals >= 0, x_vals, x_vals * negative_slope)
    
    # Vectorized store
    tl.store(output_ptr + offsets_flat, output_vals, mask=mask_flat)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'VEC_WIDTH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'VEC_WIDTH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'VEC_WIDTH': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'VEC_WIDTH': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'VEC_WIDTH': 2}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def leaky_relu_kernel_2d_optimized_vec(
    x_ptr,
    output_ptr,
    negative_slope,
    M, N,
    stride_m, stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    VEC_WIDTH: tl.constexpr,
):
    """Optimized 2D LeakyReLU kernel with vectorized access in N dimension."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Create block offsets with vectorization in N dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n_base = pid_n * BLOCK_SIZE_N * VEC_WIDTH
    offs_n_vec = offs_n_base + tl.arange(0, BLOCK_SIZE_N)[:, None] * VEC_WIDTH + tl.arange(0, VEC_WIDTH)[None, :]
    
    # Flatten N dimension offsets
    offs_n_flat = tl.reshape(offs_n_vec, (BLOCK_SIZE_N * VEC_WIDTH,))
    
    # Create masks for boundary checking
    mask_m = offs_m < M
    mask_n = offs_n_flat < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Compute memory offsets with coalesced access
    x_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n_flat[None, :] * stride_n
    
    # Load data with proper masking
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Branchless LeakyReLU
    output = tl.where(x >= 0, x, x * negative_slope)
    
    # Store with same mask
    tl.store(x_ptrs, output, mask=mask)

def triton_leaky_relu_optimized(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """High-performance LeakyReLU implementation with optimized memory access."""
    output = torch.empty_like(x)
    
    # Handle 1D and 2D contiguous tensors with vectorized kernel
    if x.ndim <= 2 and x.is_contiguous():
        n_elements = x.numel()
        
        # Use vectorized 1D kernel for contiguous memory
        grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE'] * META['VEC_WIDTH']),)
        
        leaky_relu_kernel_optimized_vec[grid](
            x, output, negative_slope, n_elements
        )
    else:
        # For non-contiguous or higher dimensional tensors, use 2D approach
        if x.ndim > 2:
            # Flatten batch dimensions
            original_shape = x.shape
            if x.ndim == 3:
                x_2d = x.view(-1, x.shape[-1])
                output_2d = output.view(-1, x.shape[-1])
            elif x.ndim == 4:
                x_2d = x.view(-1, x.shape[-2], x.shape[-1])
                output_2d = output.view(-1, x.shape[-2], x.shape[-1])
                # Reshape to 2D: (batch*M) x N
                x_2d = x_2d.view(-1, x_2d.shape[-1])
                output_2d = output_2d.view(-1, output_2d.shape[-1])
            else:
                # For >4D, flatten all but last dimension
                x_2d = x.view(-1, x.shape[-1])
                output_2d = output.view(-1, x.shape[-1])
        else:
            x_2d = x
            output_2d = output
        
        M, N = x_2d.shape[-2], x_2d.shape[-1]
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N'] * META['VEC_WIDTH']),
        )
        
        leaky_relu_kernel_2d_optimized_vec[grid](
            x_2d, output_2d, negative_slope,
            M, N,
            x_2d.stride(0) if x_2d.dim() > 1 else 0,
            x_2d.stride(-1),
        )
    
    return output

class ModelNew(nn.Module):
    """
    High-performance LeakyReLU model using optimized Triton kernels with vectorized memory access.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor using optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu_optimized(x, negative_slope=self.negative_slope)
