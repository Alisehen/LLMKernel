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
    # Tile sizes
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
    
    # Masks for boundary checks
    x_mask = (rm[:, None] < M) & (rk[None, :] < K)
    w_mask = (rn[:, None] < N) & (rk[None, :] < K)
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute number of blocks in K dimension
    K_BLOCKS = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Blocked matrix multiplication
    for k in range(K_BLOCKS):
        # Load tile from input x
        k_off = k * BLOCK_SIZE_K
        x_ptrs = x_ptr + (rm[:, None] * stride_xm + (k_off + rk[None, :]) * stride_xk)
        x_tile = tl.load(x_ptrs, mask=x_mask & ((k_off + rk[None, :]) < K), other=0.0)
        
        # Load tile from weight w
        w_ptrs = w_ptr + (rn[:, None] * stride_wn + (k_off + rk[None, :]) * stride_wk)
        w_tile = tl.load(w_ptrs, mask=w_mask & ((k_off + rk[None, :]) < K), other=0.0)
        
        # Accumulate matrix product
        acc += tl.dot(x_tile, tl.trans(w_tile))
    
    # Add bias if provided
    if b_ptr is not None:
        b_ptrs = b_ptr + rn
        b_tile = tl.load(b_ptrs, mask=rn < N, other=0.0)
        acc += b_tile[None, :]
    
    # Apply scaling and residual: output = (x @ W^T + b) * scaling_factor + (x @ W^T + b)
    # This simplifies to: output = (x @ W^T + b) * (scaling_factor + 1)
    combined_factor = scaling_factor + 1.0
    acc = acc * combined_factor
    
    # Store result
    out_ptrs = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
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
    
    # Configurable tile sizes (powers of 2 for best performance)
    # These can be tuned for specific hardware
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    # Compute grid size
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
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
        BLOCK_SIZE_K=BLOCK_SIZE_K
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
