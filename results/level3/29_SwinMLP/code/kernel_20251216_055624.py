# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton kernel: fused linear (GEMM + bias)
# -----------------------------
# Target: Ada (SM8.9, e.g., RTX 4090)
# Focus: BLOCK_SIZE tuning under register-pressure constraints:
#   - BLOCK_M/N in {32, 64}, BLOCK_K = 32
#   - Keep num_stages modest to avoid register spilling
#   - Autotune a few tilings and choose at runtime

@triton.autotune(
    configs=[
        # Balanced tile for most shapes, good reuse, moderate register usage
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        # Narrow N for very tall-and-skinny GEMMs (large M, small N)
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        # Narrow M for very wide GEMMs (small M, large N)
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_kernel(
    a_ptr,  # *const T, A: [M, K]
    b_ptr,  # *const T, B: [K, N]  (i.e., weight.T contiguous)
    bias_ptr,  # *const T, bias: [N] or dummy
    c_ptr,  # *mut T, C: [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute: C = A @ B + bias
    A: [M, K]
    B: [K, N]
    bias: [N] (optional)
    C: [M, N]
    """

    # ----------------------------------
    # Program id and grouped ordering in M to improve L2 locality
    # ----------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M
    group_id = pid // (group_size * num_pid_n)
    first_pid_m = group_id * group_size
    group_size = tl.minimum(num_pid_m - first_pid_m, group_size)
    pid_in_group = pid % (group_size * num_pid_n)
    pid_m = first_pid_m + pid_in_group // num_pid_n
    pid_n = pid_in_group % num_pid_n

    # ----------------------------------
    # Offsets
    # ----------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Make pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in fp32 for numerical stability, even if inputs are fp16/bf16
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----------------------------------
    # Main K-loop
    # ----------------------------------
    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        # Masks to handle ragged tiles in K, M, and N
        k_mask = offs_k < k_remaining
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        b_mask = k_mask[:, None] & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # tf32 acceleration on Ada for fp32 inputs; ignored for fp16/bf16
        acc += tl.dot(a, b, allow_tf32=True)

        # Bump pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # ----------------------------------
    # Bias add (epilogue)
    # ----------------------------------
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        # Keep bias separately; adds negligible register pressure
        acc += bias[None, :]

    # ----------------------------------
    # Write back
    # ----------------------------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# -----------------------------
# Wrapper: Triton fused linear
# -----------------------------

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Fused linear: y = x @ weight.T + bias

    x: [M, K]
    weight: [N, K]  (standard nn.Linear layout)
    bias: [N] or None
    """
    assert x.is_cuda and weight.is_cuda, "triton_linear only supports CUDA tensors"
    assert x.dtype == weight.dtype, "Input and weight dtypes must match"
    assert x.is_contiguous() and weight.is_contiguous(), "Inputs must be contiguous for best performance"

    M, K = x.shape
    N, K_w = weight.shape
    assert K_w == K, "Incompatible shapes for linear"

    # B = weight.T laid out as [K, N] for coalesced loads
    b_mat = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_cuda and bias.shape[0] == N
        bias_ptr = bias
    else:
        # Dummy tensor (never accessed when HAS_BIAS=False)
        bias_ptr = weight.new_empty(1)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    _linear_kernel[grid](
        x,                      # a_ptr
        b_mat,                  # b_ptr
        bias_ptr,               # bias_ptr
        y,                      # c_ptr
        M, N, K,
        x.stride(0), x.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=has_bias,
    )
    return y


# -----------------------------
# Minimal example model using Triton linear
# -----------------------------

class ModelNew(nn.Module):
    """
    Simple high-performance MLP using Triton-optimized linear layers.

    The model is intentionally minimal to highlight the fused GEMM(+bias)
    kernel performance on Ada (e.g., RTX 4090).

    Args:
        in_features  (int): input feature dimension
        hidden_features (int): hidden layer dimension
        out_features (int): output feature dimension
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        # Use nn.Parameter for weights/biases; forward uses Triton kernel
        self.w1 = nn.Parameter(torch.empty(hidden_features, in_features))
        self.b1 = nn.Parameter(torch.empty(hidden_features))
        self.w2 = nn.Parameter(torch.empty(out_features, hidden_features))
        self.b2 = nn.Parameter(torch.empty(out_features))

        # Kaiming-style init
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in1 = in_features
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.b1, -bound1, bound1)
        fan_in2 = hidden_features
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.b2, -bound2, bound2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features], CUDA tensor
        """
        assert x.is_cuda, "ModelNew expects CUDA inputs to leverage Triton kernels"

        # Layer 1: Triton fused linear + GELU
        x = triton_linear(x, self.w1, self.b1)
        x = torch.nn.functional.gelu(x)

        # Layer 2: Triton fused linear (no activation)
        x = triton_linear(x, self.w2, self.b2)
        return x
