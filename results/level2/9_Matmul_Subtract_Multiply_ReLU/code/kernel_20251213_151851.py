import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'num_warps': 4, 'num_stages': 3}),  # Original config
        triton.Config({'num_warps': 8, 'num_stages': 3}),  # Higher occupancy
        triton.Config({'num_warps': 8, 'num_stages': 2}),  # Reduced shared memory pressure
        triton.Config({'num_warps': 8, 'num_stages': 4}),  # More pipelining
        triton.Config({'num_warps': 4, 'num_stages': 2}),  # Lower shared memory
        triton.Config({'num_warps': 2, 'num_stages': 4}),  # More concurrent blocks
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_relu_kernel(
    # Pointers to matrices
    a_ptr,
    w_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # Subtractive and multiplicative factors
    subtract_value,
    multiply_value,
    # Strides
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Fused matrix multiplication with bias, subtraction, scaling, and ReLU."""
    
    # -----------------------------------------------------------
    # Map program ids to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Create block pointers for the first blocks of A and B.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Precompute mask for boundary checks
    k_mask = offs_k[None, :] < K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load with boundary mask
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        w = tl.load(w_ptrs, mask=k_mask, other=0.0)
        
        # Use tensor cores for the dot product
        accumulator += tl.dot(a, w, allow_tf32=True)
        
        # Update pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk
        # Update mask for next iteration
        k_mask = offs_k[None, :] < (K - (k + 1) * BLOCK_SIZE_K)

    # -----------------------------------------------------------
    # Add bias if provided
    if b_ptr is not None:
        b_ptrs = b_ptr + offs_bn
        bias = tl.load(b_ptrs)
        accumulator += bias[None, :]

    # -----------------------------------------------------------
    # Apply subtract, multiply, and ReLU in a fused manner
    accumulator = tl.where(accumulator > subtract_value, 
                          (accumulator - subtract_value) * multiply_value, 
                          0.0)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def triton_matmul_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    subtract_value: float,
    multiply_value: float,
) -> torch.Tensor:
    """Triton wrapper for fused matmul with bias, subtract, scale, and ReLU."""
    
    # Check constraints
    assert x.is_cuda and weight.is_cuda
    if bias is not None:
        assert bias.is_cuda
        
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2, f"Incompatible dimensions: {K} != {K2}"
    
    # Allocate output
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Ensure tensor is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    
    # 1D launch kernel where each block produces one block of C
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    fused_matmul_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        subtract_value, multiply_value,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    Optimized with Triton kernels.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Use Triton kernel for the entire fused operation
        return triton_matmul_relu(
            x,
            self.linear.weight,
            self.linear.bias,
            self.subtract_value,
            self.multiply_value,
        )
