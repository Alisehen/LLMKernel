import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_bn_gelu_relu_kernel(
    x_ptr,  # fp16 [M, K]
    w_ptr,  # fp16 [N, K]
    b_ptr,  # fp32 [N]
    running_mean_ptr,  # fp32 [N]
    running_var_ptr,   # fp32 [N]
    gamma_ptr,         # fp32 [N]
    beta_ptr,          # fp32 [N]
    y_ptr,             # fp32 [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

        # Load as fp16 to use Tensor Cores; tl.dot(fp16, fp16) accumulates in fp32
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # fp16
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # fp16

        acc += tl.dot(a, b, allow_tf32=True)  # fp32 accumulator

    # Add linear bias (broadcast over rows) in fp32
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)  # fp32
    acc += bias[None, :]

    # BatchNorm (inference): y = (x - mean) / sqrt(var + eps) * gamma + beta
    rm = tl.load(running_mean_ptr + offs_n, mask=offs_n < N, other=0.0)  # fp32
    rv = tl.load(running_var_ptr + offs_n, mask=offs_n < N, other=0.0)   # fp32
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=1.0)      # fp32
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)        # fp32

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

    # Store result (fp32)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, out, mask=y_mask)


def fused_linear_bn_gelu_relu(x, linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fused implementation of:
        y = ReLU(GELU(BatchNorm1d(Linear(x))))
    using a single Triton kernel in inference mode.

    GEMM is performed in fp16 on Tensor Cores with fp32 accumulation;
    BatchNorm, GELU, and ReLU are computed in fp32. The final output
    is fp32 to match the reference PyTorch model.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"

    # Shapes: x [M, K], weight [N, K]
    M, K = x.shape
    w = linear.weight
    N = w.shape[0]
    assert w.shape[1] == K, "Incompatible Linear weight shape"

    # Cast activations and weights to fp16 for the GEMM
    # (This is the bandwidth/compute-density optimization.)
    x_half = x.to(torch.float16).contiguous()
    w_half = w.to(torch.float16).contiguous()

    # Handle optional bias (keep in fp32)
    if linear.bias is None:
        b = torch.zeros(
            N, device=x.device, dtype=torch.float32
        )
    else:
        b = linear.bias.to(torch.float32).contiguous()

    # BatchNorm running stats and affine parameters (fp32)
    running_mean = bn.running_mean.to(torch.float32).contiguous()
    running_var = bn.running_var.to(torch.float32).contiguous()

    if bn.weight is None:
        gamma = torch.ones_like(running_mean, dtype=torch.float32, device=x.device)
    else:
        gamma = bn.weight.to(torch.float32).contiguous()

    if bn.bias is None:
        beta = torch.zeros_like(running_mean, dtype=torch.float32, device=x.device)
    else:
        beta = bn.bias.to(torch.float32).contiguous()

    eps = float(bn.eps)

    # Output in fp32 to match reference model behavior
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta["BLOCK_M"])),
            max(1, triton.cdiv(N, meta["BLOCK_N"])),
        )

    fused_linear_bn_gelu_relu_kernel[grid](
        x_half, w_half, b,
        running_mean, running_var,
        gamma, beta,
        y,
        M, N, K,
        x_half.stride(0), x_half.stride(1),
        w_half.stride(0), w_half.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.

    - In training mode or on CPU: falls back to standard PyTorch ops
      (including BatchNorm running-stat updates).
    - In eval mode on CUDA: uses a fused Triton kernel implementing
      Linear (GEMM in fp16 + fp32 accumulation) + BatchNorm (inference) + GELU + ReLU.
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
