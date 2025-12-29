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
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul: x [M, K], w [N, K], compute x @ w^T
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k  # global K indices for this tile

        a_ptrs = x_ptr + (
            offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        )  # [BLOCK_M, BLOCK_K]
        b_ptrs = w_ptr + (
            offs_n[None, :] * stride_wn + k_offsets[:, None] * stride_wk
        )  # [BLOCK_K, BLOCK_N], w[n, k]

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_offsets[:, None] < K)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Add linear bias (broadcast over rows)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # BatchNorm (inference): y = (x - mean) / sqrt(var + eps) * gamma + beta
    rm = tl.load(running_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    rv = tl.load(running_var_ptr + offs_n, mask=offs_n < N, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)

    inv_std = 1.0 / tl.sqrt(rv + eps)
    scale = gamma * inv_std
    bn_bias = beta - rm * scale

    acc = acc * scale[None, :] + bn_bias[None, :]

    # GELU (match torch.nn.functional.gelu, approximate="none"):
    # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT_2 = 0.70710678118654752440
    x_val = acc
    erf_arg = x_val * INV_SQRT_2
    gelu_val = 0.5 * x_val * (1.0 + tl.math.erf(erf_arg))

    # ReLU
    out = tl.maximum(gelu_val, 0.0)

    # Store result
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, out, mask=y_mask)


def fused_linear_bn_gelu_relu(x, linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fused implementation of:
        y = ReLU(GELU(BatchNorm1d(Linear(x))))
    using a single Triton kernel in inference mode.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"

    x = x.contiguous()
    w = linear.weight.contiguous()  # [N, K]

    # Handle optional bias
    if linear.bias is None:
        b = torch.zeros(
            w.shape[0], device=x.device, dtype=w.dtype
        )
    else:
        b = linear.bias.contiguous()

    running_mean = bn.running_mean.contiguous()
    running_var = bn.running_var.contiguous()

    # Handle optional affine parameters in BatchNorm
    if bn.weight is None:
        gamma = torch.ones_like(running_mean)
    else:
        gamma = bn.weight.contiguous()

    if bn.bias is None:
        beta = torch.zeros_like(running_mean)
    else:
        beta = bn.bias.contiguous()

    eps = bn.eps

    M, K = x.shape
    N = w.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

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

    - In training mode or on CPU: falls back to standard PyTorch ops
      (including BatchNorm running-stat updates).
    - In eval mode on CUDA: uses a fused Triton kernel implementing
      Linear + BatchNorm (inference) + GELU + ReLU.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Training or CPU: use exact PyTorch behavior
        if (not x.is_cuda) or self.training:
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x

        # Inference on CUDA: use fused Triton implementation
        return fused_linear_bn_gelu_relu(x, self.gemm, self.batch_norm)
