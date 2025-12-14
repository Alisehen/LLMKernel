import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_scale_residual_kernel(
    # Pointers to matrices
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    # Matrix dimensions
    M, N, K,
    # Scaling factor
    scaling_factor,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Tile sizes - optimized for performance
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel for linear transformation, scaling, and residual addition."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for the M and N dimensions
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute pointer offsets for x and w with proper broadcasting
    x_ptrs = x_ptr + (rm[:, None] * stride_xm + rk[None, :] * stride_xk)
    w_ptrs = w_ptr + (rn[:, None] * stride_wn + rk[None, :] * stride_wk)
    
    # Blocked matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles with boundary checking
        k_remaining = K - k
        k_mask = rk[None, :] < k_remaining
        
        x_mask = (rm[:, None] < M) & k_mask
        x_tile = tl.load(x_ptrs + k * stride_xk, mask=x_mask, other=0.0)
        
        w_mask = (rn[:, None] < N) & k_mask
        w_tile = tl.load(w_ptrs + k * stride_wk, mask=w_mask, other=0.0)
        
        # Compute matrix multiplication: x_tile * w_tile^T
        # w_tile is (BLOCK_SIZE_N, BLOCK_SIZE_K), we need (BLOCK_SIZE_K, BLOCK_SIZE_N) for dot product
        acc += tl.dot(x_tile, tl.trans(w_tile))
    
    # Add bias if provided
    if b_ptr is not None:
        b_ptrs = b_ptr + rn[None, :]
        b_mask = rn[None, :] < N
        bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += bias
    
    # Apply scaling and residual: output = (x @ W^T + b) * (scaling_factor + 1)
    combined_factor = scaling_factor + 1.0
    acc = acc * combined_factor
    
    # Store result with boundary checking
    out_ptrs = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)

def triton_fused_linear_scale_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float
) -> torch.Tensor:
    """Wrapper function for fused linear transformation, scaling, and residual addition."""
    M, K = x.shape
    N, _ = weight.shape
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Optimized tile sizes for A100/V100 (64KB shared memory limit)
    # Using 64x64x32 configuration = 64*32 + 64*32 = 4096 + 4096 = 8192 elements
    # At 4 bytes per element = 32768 bytes (well within 64KB limit)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # Compute grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel with optimized parameters
    fused_linear_scale_residual_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        scaling_factor,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=4,  # Reduced warps for better occupancy
        num_stages=3
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, scaling, and residual addition
    using fused Triton kernel.
    """
    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling_factor = scaling_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fused Triton kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return triton_fused_linear_scale_residual(
            x, self.weight, self.bias, self.scaling_factor
        )
