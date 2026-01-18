import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def int8_matmul_dequant_kernel(
    a_ptr, b_ptr, scale_x_ptr, scale_w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    INT8 matmul with fused dequantization and bias addition.
    A: [M, K] int8
    B: [K, N] int8 (transposed weight)
    scale_x: [M] float32
    scale_w: [N] float32
    bias: [N] float16
    C: [M, N] float16
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # Accumulator in INT32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        
        # Load A block as int8
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        
        # Load B block as int8
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        
        # Cast to int32 for accumulation and compute dot product
        a_i32 = a.to(tl.int32)
        b_i32 = b.to(tl.int32)
        
        # Manual matmul for int types (tl.dot doesn't support int8 directly on all hardware)
        acc += tl.dot(a_i32, b_i32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Convert accumulator to float32 for dequantization
    acc_f32 = acc.to(tl.float32)
    
    # Load scales
    scale_x = tl.load(scale_x_ptr + offs_m, mask=offs_m < M, other=0.0)
    scale_w = tl.load(scale_w_ptr + offs_n, mask=offs_n < N, other=0.0)
    
    # Dequantization factor
    divfactor = 1.0 / (127.0 * 127.0)
    
    # Apply dequantization: scale_w * scale_x * (result * divfactor)
    # scale_x: [BLOCK_M] -> [BLOCK_M, 1]
    # scale_w: [BLOCK_N] -> [1, BLOCK_N]
    output = scale_w[None, :] * (scale_x[:, None] * (acc_f32 * divfactor))
    
    # Load and add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    output = output + bias[None, :].to(tl.float32)
    
    # Convert to float16
    output_f16 = output.to(tl.float16)
    
    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, output_f16, mask=c_mask)


def int8_matmul_dequant(x, weight_int8, scale_x, scale_w, bias):
    M, K = x.shape
    N = weight_int8.shape[0]
    
    # Transpose weight: [N, K] -> [K, N]
    weight_t = weight_int8.t().contiguous()
    
    # Output tensor
    c = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Grid configuration
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    int8_matmul_dequant_kernel[grid](
        x, weight_t, scale_x, scale_w, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features=2048, out_features=2048):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight matrix (INT8)
        self.weight_int8 = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )

        # Per-column scale for weights
        self.scale_w = nn.Parameter(
            torch.randn(out_features, dtype=torch.float32).abs() * 0.01,
            requires_grad=False
        )

        # Optional bias
        self.bias = nn.Parameter(
            torch.randn(out_features, dtype=torch.float16) * 0.01,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on CUDA and contiguous
        x = x.cuda().contiguous()
        scale_x = scale_x.cuda().contiguous()
        
        return int8_matmul_dequant(
            x, 
            self.weight_int8, 
            scale_x, 
            self.scale_w, 
            self.bias
        )
