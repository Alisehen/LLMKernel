import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Primary: High occupancy with register-friendly warps
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        # Secondary: Balanced for moderate K
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        # Tensor-core optimized (multiples of 16, lower warps for register pressure)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def linear_activation_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    # 2D grid for matrix multiplication
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for the current tile with static ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pre-compute pointers for A and B
    a_ptrs_base = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs_base = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    
    # Initialize accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop with K-tiling
    for k in range(0, tl.cdiv(K, BLOCK_K) * BLOCK_K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        
        a = tl.load(a_ptrs_base, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs_base, mask=b_mask, other=0.0)
        
        # Tensor core accelerated matmul
        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32)
        
        # Increment pointers
        a_ptrs_base += BLOCK_K * stride_ak
        b_ptrs_base += BLOCK_K * stride_bk
    
    # Fused bias addition
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Fused activation with optimized implementations
    if ACTIVATION == 1:  # ReLU
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == 2:  # GELU approximation
        x = acc
        # Optimized: x * sigmoid(1.702 * x)
        sigmoid_input = x * 1.702
        sigmoid = 1.0 / (1.0 + tl.exp(-sigmoid_input))
        acc = x * sigmoid
    elif ACTIVATION == 3:  # SiLU
        x = acc
        sigmoid = 1.0 / (1.0 + tl.exp(-x))
        acc = x * sigmoid
    
    # Store final output with coalesced writes
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def fused_linear_activation(x, weight, bias=None, activation='none'):
    M, K = x.shape
    N = weight.shape[0]
    
    # Ensure contiguous memory layout
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Pre-allocate output tensor
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid calculation
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    # Map activation string to kernel constant
    if activation == 'relu':
        ACTIVATION = 1
    elif activation == 'gelu':
        ACTIVATION = 2
    elif activation == 'silu':
        ACTIVATION = 3
    else:
        ACTIVATION = 0
    
    # Launch kernel
    linear_activation_kernel[grid](
        x, weight, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=ACTIVATION,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32
    )
    
    return c

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation='relu'):
        super(ModelNew, self).__init__()
        
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        self.output_layer = nn.Linear(current_input_size, output_size)
        self.activation = activation
    
    def forward(self, x):
        # Hidden layers with activation
        for layer in self.layers:
            x = fused_linear_activation(x, layer.weight, layer.bias, activation=self.activation)
        
        # Output layer without activation
        x = fused_linear_activation(x, self.output_layer.weight, self.output_layer.bias, activation='none')
        
        return x
