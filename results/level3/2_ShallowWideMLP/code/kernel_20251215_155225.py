import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Optimized for register pressure and Ada Lovelace
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        # Smaller blocks for high register pressure cases
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Tensor-core optimized with moderate register usage
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
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
    
    # Offsets with masking for boundaries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for A and B tiles with precomputation
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main matmul loop with K-tiling
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        
        # Load with boundary checks
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate matrix product
        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32)
        
        # Move pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Fused bias addition (recomputed in activation to save registers)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Fused activation with register-optimized implementations
    if ACTIVATION == 1:  # ReLU
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == 2:  # GELU approximation
        # Optimized GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = acc
        x3 = x * x * x
        inner = 0.7978845608 * (x + 0.044715 * x3)  # sqrt(2/pi) ≈ 0.7978845608
        tanh_val = (2.0 / (1.0 + tl.exp(-2.0 * inner))) - 1.0  # tanh approximation
        acc = 0.5 * x * (1.0 + tanh_val)
    elif ACTIVATION == 3:  # SiLU (Swish)
        # Optimized SiLU: x * sigmoid(x)
        x = acc
        sigmoid = 1.0 / (1.0 + tl.exp(-x))
        acc = x * sigmoid
    
    # Store output with boundary check
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def fused_linear_activation(x, weight, bias=None, activation='none'):
    M, K = x.shape
    N = weight.shape[0]
    
    # Ensure contiguous memory
    x = x.contiguous()
    weight = weight.contiguous()
    
    # Pre-allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Grid calculation
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    # Activation mapping
    activation_map = {'relu': 1, 'gelu': 2, 'silu': 3}
    ACTIVATION = activation_map.get(activation, 0)
    
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
