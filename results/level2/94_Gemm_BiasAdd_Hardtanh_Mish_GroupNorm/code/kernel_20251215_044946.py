import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_bias_hardtanh_mish_kernel(
    a_ptr, b_ptr,
    bias0_ptr, bias1_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel:
      C = mish(hardtanh(A @ B + bias0 + bias1))

    A: [M, K]
    B: [K, N]  (transposed weight)
    bias0: [N]  (linear bias)
    bias1: [N]  (extra bias added in the model)
    C: [M, N]

    Grid:
      program_id(0) -> blocks along M
      program_id(1) -> blocks along N

    All post-GEMM fused ops (bias add, Hardtanh, Mish, store)
    use the SAME (offs_m, offs_n) and masks derived from them.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile coordinates
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]
    offs_k = tl.arange(0, BLOCK_K)                    # [BK]

    # Hints for compiler (help vectorization / pipelining)
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Pointers for the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_mask[None, :]),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None]) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Use tensor cores (TF32) for maximum throughput on Ada
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # ----- Fused epilogue: bias + Hardtanh + Mish -----
    # Unified offsets and masks for all fused elementwise ops
    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Bias loads use the SAME offs_n and mask_n as the output tile
    bias0 = tl.load(bias0_ptr + offs_n, mask=mask_n, other=0.0)
    bias1 = tl.load(bias1_ptr + offs_n, mask=mask_n, other=0.0)
    bias = bias0 + bias1

    # Broadcast bias across M for this tile
    acc = acc + bias[None, :]

    # Hardtanh: clamp to [-1, 1]
    acc = tl.maximum(acc, -1.0)
    acc = tl.minimum(acc, 1.0)

    # Mish: x * tanh(softplus(x)), with numerically-stable softplus
    abs_x = tl.abs(acc)
    softplus = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-abs_x))

    # tanh(softplus) via exp: tanh(z) = (e^{2z} - 1) / (e^{2z} + 1)
    e2 = tl.exp(2.0 * softplus)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)

    acc = acc * tanh_sp

    # Store result with the SAME (offs_m, offs_n) and unified mask
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_gemm_bias_hardtanh_mish(x, weight, bias_linear, bias_extra):
    """
    x: [M, K]
    weight: [N, K]
    bias_linear: [N]
    bias_extra: [N]
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, "in_features mismatch"

    # Output
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Use transposed weight for better memory access: [K, N] contiguous
    # Compute in fp32 even if inputs are lower precision
    a = x.to(torch.float32)
    b = weight.to(torch.float32).t().contiguous()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    # Tuned for Ada (4090): larger tiles + more warps + deeper pipeline
    gemm_bias_hardtanh_mish_kernel[grid](
        a, b,
        bias_linear, bias_extra,
        c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=4,
    )

    # Match original input dtype on exit
    return c.to(x.dtype)


@triton.jit
def groupnorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    N, C, G,
    stride_xn, stride_xc,
    stride_yn, stride_yc,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    GroupNorm over (N, C) with G groups, group size = GROUP_SIZE = C // G.

    For each (n, g):
      - Compute mean / var over channels [g*GROUP_SIZE : (g+1)*GROUP_SIZE]
      - y[n, c] = gamma[c] * (x[n, c] - mean) / sqrt(var + eps) + beta[c]
    """

    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    n = pid_n
    g = pid_g

    c_start = g * GROUP_SIZE

    # First pass: compute sum and sum of squares (scalar accumulators)
    mean = 0.0
    m2 = 0.0

    # Loop over channels in this group, BLOCK_SIZE at a time
    off = 0
    while off < GROUP_SIZE:
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = (n < N) & (g < G) & (offs < GROUP_SIZE)
        c = c_start + offs

        x_ptrs = x_ptr + n * stride_xn + c * stride_xc
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        mean += tl.sum(x, axis=0)
        m2 += tl.sum(x * x, axis=0)

        off += BLOCK_SIZE

    inv_size = 1.0 / GROUP_SIZE
    mean = mean * inv_size
    var = m2 * inv_size - mean * mean
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, scale, shift
    off = 0
    while off < GROUP_SIZE:
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = (n < N) & (g < G) & (offs < GROUP_SIZE)
        c = c_start + offs

        x_ptrs = x_ptr + n * stride_xn + c * stride_xc
        y_ptrs = y_ptr + n * stride_yn + c * stride_yc

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0)
        beta = tl.load(beta_ptr + c, mask=mask, other=0.0)

        y = (x - mean) * rstd
        y = y * gamma + beta

        tl.store(y_ptrs, y, mask=mask)

        off += BLOCK_SIZE


def groupnorm_triton(x, weight, bias, num_groups, eps=1e-5):
    """
    x: [N, C] (contiguous)
    weight: [C]
    bias: [C]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.ndim == 2
    N, C = x.shape
    assert C % num_groups == 0, "num_channels must be divisible by num_groups"
    group_size = C // num_groups

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    def grid(meta):
        return (N, num_groups)

    groupnorm_kernel[grid](
        x_contig, weight, bias, y,
        N, C, num_groups,
        x_contig.stride(0), x_contig.stride(1),
        y.stride(0), y.stride(1),
        eps,
        GROUP_SIZE=group_size,
        BLOCK_SIZE=128,
        num_warps=4,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for the reference model:

      x -> Linear -> +bias -> Hardtanh -> Mish -> GroupNorm

    The parameter/module structure mirrors the reference:
      - self.linear: nn.Linear(in_features, out_features)
      - self.bias:   nn.Parameter(bias_shape)
      - self.norm:   nn.GroupNorm(num_groups, out_features)
    """

    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.eps = 1e-5

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bias = nn.Parameter(torch.empty(bias_shape))
        self.norm = nn.GroupNorm(num_groups, out_features, eps=self.eps, affine=True)

    def forward(self, x):
        # Fused GEMM + bias + Hardtanh + Mish (Triton)
        y = fused_gemm_bias_hardtanh_mish(
            x,
            self.linear.weight,
            self.linear.bias,
            self.bias,
        )
        # GroupNorm (Triton) with same affine params as nn.GroupNorm
        y = groupnorm_triton(
            y,
            self.norm.weight,
            self.norm.bias,
            self.num_groups,
            self.eps,
        )
        return y
