import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def int8_matmul_dequant_kernel(
    # Pointers
    x_ptr, weight_t_ptr, scale_x_ptr, scale_w_ptr, bias_ptr, output_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_om, stride_on,
    # Constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused INT8 matmul with dequantization kernel - optimized with pre-transposed weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Pointers for x [M, K] and weight_t [K, N] (pre-transposed)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight_t is [K, N], we load [BLOCK_K, BLOCK_N] directly
    w_ptrs = weight_t_ptr + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
    
    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        
        # Load x block [BLOCK_M, BLOCK_K]
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        x_block = tl.load(x_ptrs, mask=mask_x, other=0)
        x_float = x_block.to(tl.float32)
        
        # Load weight block [BLOCK_K, BLOCK_N] directly (no transpose needed)
        mask_w = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        w_block = tl.load(w_ptrs, mask=mask_w, other=0)
        w_float = w_block.to(tl.float32)
        
        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_float, w_float, allow_tf32=True)
        
        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wtk
    
    # Apply dequantization
    divfactor = 1.0 / (127.0 * 127.0)
    acc = acc * divfactor
    
    # Load and apply scale_x [M] -> [BLOCK_M, 1]
    scale_x = tl.load(scale_x_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc = acc * scale_x[:, None]
    
    # Load and apply scale_w [N] -> [1, BLOCK_N]
    scale_w = tl.load(scale_w_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale_w[None, :]
    
    # Load and add bias [N] -> [1, BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    # Convert to float16 and store
    output = acc.to(tl.float16)
    
    # Store output
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, output, mask=mask_out)


def int8_matmul_dequant(x, weight_t, scale_x, scale_w, bias):
    """
    Perform INT8 matmul with dequantization using pre-transposed weights.
    
    Args:
        x: Input tensor [M, K], dtype=int8
        weight_t: Pre-transposed weight tensor [K, N], dtype=int8
        scale_x: Per-row scale [M], dtype=float32
        scale_w: Per-column scale [N], dtype=float32
        bias: Bias tensor [N], dtype=float16
    
    Returns:
        Output tensor [M, N], dtype=float16
    """
    M, K = x.shape
    K_w, N = weight_t.shape
    
    output = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    int8_matmul_dequant_kernel[grid](
        x, weight_t, scale_x, scale_w, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return output


class ModelNew(nn.Module):
    """
    INT8 MatMul with Row-wise Dequantization - Optimized Triton Implementation
    
    Optimization: Pre-transpose weight matrix to [K, N] layout to eliminate
    per-iteration transpose in the kernel and improve memory coalescing.
    """
    def __init__(self, in_features=2048, out_features=2048):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Original weight matrix [N, K]
        weight_int8 = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
        
        # Pre-transpose weight to [K, N] for optimized memory access
        self.weight_t = nn.Parameter(
            weight_int8.t().contiguous(),
            requires_grad=False
        )

        self.scale_w = nn.Parameter(
            torch.randn(out_features, dtype=torch.float32).abs() * 0.01,
            requires_grad=False
        )

        self.bias = nn.Parameter(
            torch.randn(out_features, dtype=torch.float16) * 0.01,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        """
        Perform INT8 matrix multiplication with dequantization.

        Args:
            x (torch.Tensor): Quantized input of shape (M, K), dtype=int8
            scale_x (torch.Tensor): Per-row scale for input of shape (M,), dtype=float32

        Returns:
            torch.Tensor: Dequantized output of shape (M, out_features), dtype=float16
        """
        # Ensure inputs are on CUDA and contiguous
        x = x.cuda().contiguous()
        scale_x = scale_x.cuda().contiguous()
        
        return int8_matmul_dequant(
            x, 
            self.weight_t,  # Pre-transposed weight [K, N]
            scale_x, 
            self.scale_w, 
            self.bias
        )
