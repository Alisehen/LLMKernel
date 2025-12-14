import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    x_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized LeakyReLU kernel using elementwise operations with fused load/store."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute LeakyReLU: max(x, negative_slope * x)
    # Use tl.where for branchless computation - critical for performance
    negative_output = x * negative_slope
    output = tl.where(x >= 0, x, negative_output)
    
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def leaky_relu_kernel_2d(
    x_ptr,
    output_ptr,
    negative_slope,
    M, N,
    stride_m, stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Optimized 2D LeakyReLU kernel for better memory coalescing."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Create offsets for the entire block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks for both dimensions
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load data efficiently with broadcasting
    x_ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    mask = mask_m[:, None] & mask_n[None, :]
    
    x = tl.load(x_ptrs, mask=mask)
    
    # Compute LeakyReLU
    negative_output = x * negative_slope
    output = tl.where(x >= 0, x, negative_output)
    
    # Store with same mask
    tl.store(x_ptrs, output, mask=mask)

def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """High-performance LeakyReLU implementation using Triton."""
    output = torch.empty_like(x)
    
    # Choose optimal strategy based on tensor size and dimensions
    if x.ndim == 1 or x.is_contiguous():
        # Use 1D kernel for contiguous or 1D tensors
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Dynamically adjust block size based on problem size
        if n_elements < 8192:
            BLOCK_SIZE = 128
        elif n_elements < 65536:
            BLOCK_SIZE = 256
        elif n_elements < 262144:
            BLOCK_SIZE = 512
        else:
            BLOCK_SIZE = 1024
        
        leaky_relu_kernel[grid](
            x, output, negative_slope, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # Use 2D kernel for better memory access patterns on 2D+ tensors
        M, N = x.shape[-2], x.shape[-1]
        
        # Heuristically determine optimal block sizes
        BLOCK_SIZE_M = 64 if M >= 64 else triton.next_power_of_2(M)
        BLOCK_SIZE_N = 64 if N >= 64 else triton.next_power_of_2(N)
        
        grid = (
            triton.cdiv(M, BLOCK_SIZE_M),
            triton.cdiv(N, BLOCK_SIZE_N),
        )
        
        leaky_relu_kernel_2d[grid](
            x, output, negative_slope,
            M, N,
            x.stride(-2) if x.ndim > 1 else 1,
            x.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
    
    return output

class ModelNew(nn.Module):
    """
    High-performance LeakyReLU model using Triton kernels.
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
        Applies LeakyReLU activation to the input tensor using Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        return triton_leaky_relu(x, negative_slope=self.negative_slope)
