# Optimized Triton code for fused Linear + BatchNorm + GELU + ReLU on RTX 4090

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline / conservative config (multi-input friendly)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider M tile – good when M is large (reuses weights more)
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider N tile – good when N is large (reuses activations more)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Aggressive config – for large, well-behaved sizes with low reg pressure
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_linear_bn_gelu_relu_kernel(
    x_ptr,          # fp16 [M, K]
    wt_ptr,         # fp16 [K, N] (pre-transposed)
    b_ptr,          # fp32 [N]
    bn_scale_ptr,   # fp32 [N] (gamma * rsqrt(var + eps))
    bn_bias_ptr,    # fp32 [N] (beta - mean * scale)
    y_ptr,          # fp32 [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 2D tiling over output matrix Y: [M, N]
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < N
    y_mask = mask_m[:, None] & mask_n[None, :]

    # Output pointers for the tile
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    # -------------------------------------------------------------------------
    # GEMM: X[M,K] @ W^T[K,N]  (fp16 inputs, fp32 accumulate)
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # X tile: [BLOCK_M, BLOCK_K]
        a_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        a_mask = mask_m[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W^T tile: [BLOCK_K, BLOCK_N]
        b_ptrs = wt_ptr + k_idx[:, None] * stride_wtk + offs_n[None, :] * stride_wtn
        b_mask = k_mask[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Tensor Core-accelerated dot-product
        acc += tl.dot(a, b)

    # -------------------------------------------------------------------------
    # Linear bias + precomputed BatchNorm (scale + bias), per-column along N
    # All kept in registers, no intermediate stores
    # -------------------------------------------------------------------------
    col_mask = mask_n

    bias = tl.load(b_ptr + offs_n, mask=col_mask, other=0.0)          # [BLOCK_N]
    scale = tl.load(bn_scale_ptr + offs_n, mask=col_mask, other=1.0)  # [BLOCK_N]
    bn_bias = tl.load(bn_bias_ptr + offs_n, mask=col_mask, other=0.0) # [BLOCK_N]

    # acc = (acc + bias) * scale + bn_bias
    acc += bias[None, :]
    acc = acc * scale[None, :] + bn_bias[None, :]

    # -------------------------------------------------------------------------
    # Exact GELU (erf-based) + ReLU, all in-register
    # -------------------------------------------------------------------------
    INV_SQRT_2 = 0.70710678118654752440  # 1 / sqrt(2)

    x_val = acc
    erf_arg = x_val * INV_SQRT_2
    erf_val = tl.math.erf(erf_arg)
    gelu_val = 0.5 * x_val * (1.0 + erf_val)

    out = tl.maximum(gelu_val, 0.0)

    # -------------------------------------------------------------------------
    # Single store of final result tile
    # -------------------------------------------------------------------------
    tl.store(y_ptrs, out, mask=y_mask)


def fused_linear_bn_gelu_relu(x, linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Fused inference implementation of:
        y = ReLU(GELU(BatchNorm1d(Linear(x))))

    Optimizations for RTX 4090:
    - GEMM in fp16 with fp32 accumulation, using Tensor Cores via tl.dot
    - Weight pre-transposed to [K, N] for coalesced loads
    - BatchNorm parameters pre-folded into scale + bias to reduce math
    - GELU (exact, erf-based) + ReLU fused in epilogue
    - Single global store (no intermediate writes)
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"

    # Shapes: x [M, K], weight [N, K]
    M, K = x.shape
    w = linear.weight
    N = w.shape[0]
    assert w.shape[1] == K, "Incompatible Linear weight shape"

    device = x.device

    # Ensure activations are fp16 and contiguous
    if x.dtype != torch.float16 or not x.is_contiguous():
        x_half = x.to(dtype=torch.float16, non_blocking=True).contiguous()
    else:
        x_half = x

    # Ensure weights are fp16 and contiguous
    if w.dtype != torch.float16 or not w.is_contiguous():
        w_half = w.to(dtype=torch.float16, device=device, non_blocking=True).contiguous()
    else:
        w_half = w

    # Pre-transpose weight to [K, N] for better memory access in kernel
    w_half_t = w_half.t().contiguous()  # [K, N]

    # Bias (fp32)
    if linear.bias is None:
        b = torch.zeros(N, device=device, dtype=torch.float32)
    else:
        b = linear.bias.to(dtype=torch.float32, device=device, non_blocking=True).contiguous()

    # BatchNorm parameters (fp32), precompute scale & bias:
    #   scale   = gamma * rsqrt(running_var + eps)
    #   bn_bias = beta - running_mean * scale
    running_mean = bn.running_mean.to(dtype=torch.float32, device=device, non_blocking=True)
    running_var = bn.running_var.to(dtype=torch.float32, device=device, non_blocking=True)

    if bn.weight is None:
        gamma = torch.ones_like(running_mean, dtype=torch.float32, device=device)
    else:
        gamma = bn.weight.to(dtype=torch.float32, device=device, non_blocking=True)

    if bn.bias is None:
        beta = torch.zeros_like(running_mean, dtype=torch.float32, device=device)
    else:
        beta = bn.bias.to(dtype=torch.float32, device=device, non_blocking=True)

    eps = float(bn.eps)

    inv_std = torch.rsqrt(running_var + eps)
    bn_scale = (gamma * inv_std).contiguous()
    bn_bias = (beta - running_mean * bn_scale).contiguous()

    # Output in fp32
    y = torch.empty((M, N), device=device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_bn_gelu_relu_kernel[grid](
        x_half,
        w_half_t,
        b,
        bn_scale,
        bn_bias,
        y,
        M,
        N,
        K,
        x_half.stride(0),
        x_half.stride(1),
        w_half_t.stride(0),  # stride_wtk
        w_half_t.stride(1),  # stride_wtn
        y.stride(0),
        y.stride(1),
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized model:

    - Training / CPU: standard PyTorch Linear + BatchNorm1d + GELU + ReLU
      (exact PyTorch behavior)
    - Eval on CUDA: aggressively optimized fused Triton kernel
      with in-register fusion and single global write.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # Training or CPU: use standard PyTorch ops for full autograd support
        if (not x.is_cuda) or self.training:
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)  # exact (erf-based) GELU by default
            x = torch.relu(x)
            return x

        # Inference on CUDA: use fused Triton implementation
        return fused_linear_bn_gelu_relu(x, self.gemm, self.batch_norm)
