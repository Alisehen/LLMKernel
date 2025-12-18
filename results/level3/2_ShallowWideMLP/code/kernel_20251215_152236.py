import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def fused_linear_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, 
                   mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), 
                   other=0.0)
        b = tl.load(b_ptrs,
                   mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
                   other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    if ACTIVATION == 1:  # ReLU
        acc = tl.maximum(acc, 0.0)
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, 
                   mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), 
                   other=0.0)
        b = tl.load(b_ptrs,
                   mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
                   other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def fused_linear_activation(x, weight, bias=None, activation='none'):
    M, K = x.shape
    N = weight.shape[0]
    
    x = x.contiguous()
    weight_t = weight.t().contiguous()
    bias = bias.contiguous() if bias is not None else None
    
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    if activation == 'relu':
        fused_linear_relu_kernel[grid](
            x, weight_t, bias, c,
            M, N, K,
            x.stride(0), x.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            c.stride(0), c.stride(1),
            ACTIVATION=1,
            ALLOW_TF32=True
        )
    else:
        linear_kernel[grid](
            x, weight_t, bias, c,
            M, N, K,
            x.stride(0), x.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            c.stride(0), c.stride(1),
            ALLOW_TF32=True
        )
    
    return c

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        self.output_layer = nn.Linear(current_input_size, output_size)
        
        for layer in self.layers:
            layer.weight.data = layer.weight.data.contiguous()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.contiguous()
        
        self.output_layer.weight.data = self.output_layer.weight.data.contiguous()
        if self.output_layer.bias is not None:
            self.output_layer.bias.data = self.output_layer.bias.data.contiguous()
    
    def forward(self, x):
        for layer in self.layers:
            x = fused_linear_activation(x, layer.weight, layer.bias, activation='relu')
        
        x = fused_linear_activation(x, self.output_layer.weight, self.output_layer.bias, activation='none')
        
        return x
