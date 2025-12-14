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
    # Tile sizes - reduced to fit shared memory
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel for linear transformation, scaling, and residual addition."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create offsets for the M and N dimensions with boundary checking
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Blocked matrix multiplication
    K_BLOCKS = tl.cdiv(K, BLOCK_SIZE_K)
    
    for k in range(K_BLOCKS):
        k_off = k * BLOCK_SIZE_K
        
        # Load tile from input x with proper boundary checking
        x_ptrs = x_ptr + (rm[:, None] * stride_xm + (k_off + rk[None, :]) * stride_xk)
        x_mask = (rm[:, None] < M) & ((k_off + rk[None, :]) < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load tile from weight w with proper boundary checking
        w_ptrs = w_ptr + (rn[:, None] * stride_wn + (k_off + rk[None, :]) * stride_wk)
        w_mask = (rn[:, None] < N) & ((k_off + rk[None, :]) < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate matrix product - transpose w_tile correctly
        acc += tl.dot(x_tile, w_tile, trans_b=True)
    
    # Add bias if provided
    if b_ptr is not None:
        b_ptrs = b_ptr + rn[None, :]
        b_mask = rn[None, :] < N
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += b_tile
    
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
    
    # Reduced tile sizes to fit within shared memory limits
    # Original: 128x128x64 = 131072 bytes (exceeds limit 101376)
    # New: 64x64x64 = 81920 bytes (within limit)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64
    
    # Compute grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
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
        num_warps=8,
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
