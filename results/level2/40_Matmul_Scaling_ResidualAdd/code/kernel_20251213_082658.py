import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_scale_residual_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    M, N, K,
    scaling_factor,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kernel for linear transformation, scaling, and residual addition."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Pre-calculate combined factor
    combined_factor = scaling_factor + 1.0
    
    # Offsets for the M and N dimensions
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Mask for rows and columns
    m_mask = rm < M
    n_mask = rn < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Blocked K-loop
    for k in range(0, K, BLOCK_K):
        # Calculate remaining K and mask
        k_remaining = K - k
        rk = tl.arange(0, BLOCK_K)
        k_mask = rk < k_remaining
        
        # Load x tile
        x_ptrs = x_ptr + (rm[:, None] * stride_xm + (k + rk[None, :]) * stride_xk)
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load w tile
        w_ptrs = w_ptr + (rn[:, None] * stride_wn + (k + rk[None, :]) * stride_wk)
        w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Accumulate matrix product
        acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=True)
    
    # Add bias if provided
    if b_ptr is not None:
        b_ptrs = b_ptr + rn[None, :]
        bias = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        acc += bias
    
    # Apply scaling factor
    acc = acc * combined_factor
    
    # Store result with boundary checking
    out_ptrs = output_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    out_mask = m_mask[:, None] & n_mask[None, :]
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
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel with autotuned parameters
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_linear_scale_residual_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        scaling_factor,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, scaling, and residual addition
    using fused Triton kernel with shared memory.
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
