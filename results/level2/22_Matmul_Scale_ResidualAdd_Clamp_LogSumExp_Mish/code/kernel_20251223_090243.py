import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def linear_scale_clamp_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale, clamp_min, clamp_max,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel:
      C = clamp( (A @ B) + bias, clamp_min, clamp_max ) * scale

    A: [M, K]
    B: [K, N]  (weight^T, i.e., transposed weight)
    bias: [N]
    C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Scale (includes the residual x + x fused as factor 2*scale_factor)
    acc = acc * scale

    # Clamp
    acc = tl.maximum(acc, clamp_min)
    acc = tl.minimum(acc, clamp_max)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_store = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_store)


@triton.jit
def logsumexp_mish_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    For each row i:
      y_i = logsumexp(x_i, dim=1)  (scalar)
      out_i = y_i * mish(y_i)

    mish(z) = z * tanh(softplus(z))
    softplus(z) = log(1 + exp(z))
    tanh(u) = (exp(2u) - 1) / (exp(2u) + 1)
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Initialize running max and sum for logsumexp
    neg_large = -1.0e30
    row_max = tl.full((BLOCK_M,), neg_large, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=neg_large,
        )

        # Current tile max per row
        tile_max = tl.max(x, axis=1)
        new_max = tl.maximum(row_max, tile_max)

        # Sum of exp(x - new_max) over this tile
        x_shifted = x - new_max[:, None]
        exp_x = tl.exp(x_shifted)
        tile_sum = tl.sum(exp_x, axis=1)

        # Update running sum with scaling to keep stability
        scale_old = tl.exp(row_max - new_max)
        row_sum = row_sum * scale_old + tile_sum
        row_max = new_max

    # Final logsumexp per row
    lse = row_max + tl.log(row_sum)

    # Mish activation on lse
    exp_lse = tl.exp(lse)
    softplus = tl.log(1.0 + exp_lse)
    twice_sp = 2.0 * softplus
    exp_twice_sp = tl.exp(twice_sp)
    tanh_sp = (exp_twice_sp - 1.0) / (exp_twice_sp + 1.0)
    mish_lse = lse * tanh_sp

    out_val = lse * mish_lse

    out_ptrs = out_ptr + offs_m * stride_outm  # single-column output
    tl.store(out_ptrs, out_val, mask=mask_m)


def fused_linear_scale_clamp(x, weight, bias, scale_total, clamp_min, clamp_max):
    """
    x: [M, K]
    weight: [N, K]  (PyTorch Linear weight: [out_features, in_features])
    bias: [N]
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be on CUDA"

    M, K = x.shape
    N = weight.shape[0]

    # B is weight^T: [K, N]
    b = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_scale_clamp_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scale_total, clamp_min, clamp_max,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    return c


def fused_logsumexp_mish(x):
    """
    x: [M, N]
    Returns: [M, 1] with y * mish(y), where y = logsumexp(x, dim=1, keepdim=True)
    """
    assert x.is_cuda, "Input must be on CUDA"

    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    logsumexp_mish_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=128,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model:
      - Fused Linear (matmul + bias) + scale + residual + clamp
      - Row-wise logsumexp
      - Mish activation and final multiply
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Initialize like nn.Linear for similar statistics
        linear = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())

        # Fused scale factor (x * scale_factor; x + x  -> x * (2 * scale_factor))
        self.scale_total = 2.0 * float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        # Expect x on CUDA for Triton kernels
        if not x.is_cuda:
            x = x.cuda()

        # Fused linear + scale + residual + clamp
        x = fused_linear_scale_clamp(
            x, self.weight, self.bias,
            self.scale_total, self.clamp_min, self.clamp_max
        )

        # Fused logsumexp + mish + final multiply
        x = fused_logsumexp_mish(x)
        return x
