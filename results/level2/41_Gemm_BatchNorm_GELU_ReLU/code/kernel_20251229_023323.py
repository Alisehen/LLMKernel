import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_bn_gelu_relu_kernel(
    x_ptr, w_ptr, b_ptr,
    running_mean_ptr, running_var_ptr,
    gamma_ptr, beta_ptr,
    y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)  # along batch dimension M
    pid_n = tl.program_id(1)  # along feature dimension N

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Pointers to blocks of A (x) and B (w) for matmul
    # x: [M, K] with strides (stride_xm, stride_xk)
    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # w: [N, K] but we access as [K, N] via strides (stride_wk, stride_wn)
    b_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator in fp32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_wk

    # Add linear bias (broadcast over rows)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # BatchNorm in inference form using running stats:
    # y = (x - mean) / sqrt(var + eps) * gamma + beta
    rm = tl.load(running_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    rv = tl.load(running_var_ptr + offs_n, mask=offs_n < N, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)

    var_eps = rv + eps
    inv_std = 1.0 / tl.sqrt(var_eps)
    scale = gamma * inv_std
    bn_bias = beta - rm * scale

    acc = acc * scale[None, :] + bn_bias[None, :]

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    k_g = 0.044715

    x_val = acc
    x2 = x_val * x_val
    x3 = x2 * x_val
    inner = c * (x_val + k_g * x3)
    two_inner = 2.0 * inner
    exp_2inner = tl.exp(two_inner)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    acc = 0.5 * x_val * (1.0 + tanh_inner)

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store result
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


def fused_linear_bn_gelu_relu(x, linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fused high-performance implementation of:
        y = ReLU(GELU(BatchNorm1d(Linear(x))))
    using a single Triton kernel in inference mode (BatchNorm uses running stats).

    Args:
        x: [M, K] input tensor on CUDA
        linear: nn.Linear(in_features=K, out_features=N)
        bn: nn.BatchNorm1d(num_features=N)

    Returns:
        y: [M, N] tensor
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    # Ensure contiguous for predictable strides
    x = x.contiguous()
    w = linear.weight.contiguous()  # [N, K]
    b = linear.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    M, K = x.shape
    N = w.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Compute grid: one kernel instance per (BLOCK_M, BLOCK_N) tile
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_bn_gelu_relu_kernel[grid](
        x, w, b,
        running_mean, running_var,
        gamma, beta,
        y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.

    - In training mode or on CPU: falls back to standard PyTorch ops for full correctness
      (including BatchNorm running-stat updates).
    - In eval mode on CUDA: uses a single fused Triton kernel implementing
      Linear + BatchNorm (inference) + GELU + ReLU for maximum performance.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Keep the same modules so state_dict is compatible with the original Model
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Training or CPU: use exact PyTorch behavior (including BN statistics)
        if (not x.is_cuda) or self.training:
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x

        # Inference on CUDA: use fused Triton implementation
        return fused_linear_bn_gelu_relu(x, self.gemm, self.batch_norm)
