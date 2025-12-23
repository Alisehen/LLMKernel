import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_linear_logsumexp_mish_kernel(
    a_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,  # strides for logical B^T: [K, N]
    stride_outm,
    scale, clamp_min, clamp_max,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused kernel performing:
      1. Linear:  A[M, K] @ W^T[N, K] + bias[N]
      2. Scale + residual: * (2 * scale_factor)  (pre-fused into `scale`)
      3. Clamp
      4. Row-wise logsumexp over N
      5. Mish activation on logsumexp result
      6. Final multiply: y * mish(y)

    Inputs:
      A: [M, K]
      W: [N, K]   (PyTorch Linear weight: [out_features, in_features])
      bias: [N]
    Output:
      out: [M, 1]
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Initialize running max and sum for logsumexp across N for each row in this block
    neg_large = -1.0e30
    row_max = tl.full((BLOCK_M,), neg_large, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over columns N in tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # GEMM tile accumulator for C[offs_m, offs_n]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-dimension tiling
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # Treat W as W^T with logical shape [K, N]:
        # element (k, n) of W^T is weight[n, k] -> stride_bk (k) and stride_bn (n)
        b_ptrs = w_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        for k in range(0, K, BLOCK_K):
            k_mask = offs_k[None, :] < (K - k)

            a = tl.load(
                a_ptrs,
                mask=(mask_m[:, None] & k_mask),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(k_mask.T & mask_n[None, :]),
                other=0.0,
            )

            acc += tl.dot(a, b, allow_tf32=True)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Add bias for this N-tile
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

        # Scale (includes x + x fused as factor 2 * scale_factor)
        acc = acc * scale

        # Clamp
        acc = tl.maximum(acc, clamp_min)
        acc = tl.minimum(acc, clamp_max)

        # For out-of-bounds columns, set to a very negative value so they don't affect logsumexp
        x = tl.where(mask_n[None, :], acc, neg_large)

        # Online logsumexp update over this tile
        tile_max = tl.max(x, axis=1)
        new_max = tl.maximum(row_max, tile_max)

        x_shifted = x - new_max[:, None]
        exp_x = tl.exp(x_shifted)
        tile_sum = tl.sum(exp_x, axis=1)

        scale_old = tl.exp(row_max - new_max)
        row_sum = row_sum * scale_old + tile_sum
        row_max = new_max

    # Final logsumexp per row
    # Avoid log(0) for inactive rows (not strictly necessary because we mask on store,
    # but keeps numerics well-defined)
    row_sum = tl.where(mask_m, row_sum, 1.0)
    lse = row_max + tl.log(row_sum)

    # Mish activation on lse:
    #   mish(z) = z * tanh(softplus(z))
    #   softplus(z) = log(1 + exp(z))
    exp_lse = tl.exp(lse)
    softplus = tl.log(1.0 + exp_lse)
    twice_sp = 2.0 * softplus
    exp_twice_sp = tl.exp(twice_sp)
    tanh_sp = (exp_twice_sp - 1.0) / (exp_twice_sp + 1.0)
    mish_lse = lse * tanh_sp

    out_val = lse * mish_lse  # y * mish(y)

    # Store result as a single column [M, 1]
    out_ptrs = out_ptr + offs_m * stride_outm
    tl.store(out_ptrs, out_val, mask=mask_m)


def fused_linear_logsumexp_mish(x, weight, bias, scale_total, clamp_min, clamp_max):
    """
    Fused operation equivalent to:

        y = x @ weight.T + bias
        y = y * scale_factor
        y = y + y
        y = clamp(y, clamp_min, clamp_max)
        y = logsumexp(y, dim=1, keepdim=True)
        out = y * mish(y)

    Args:
        x:      [M, K], input
        weight: [N, K], nn.Linear weight
        bias:   [N]
    Returns:
        out: [M, 1]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be on CUDA"

    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    fused_linear_logsumexp_mish_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # stride_bk, stride_bn for logical W^T[K, N]
        out.stride(0),
        float(scale_total), float(clamp_min), float(clamp_max),
        BLOCK_M=16, BLOCK_N=128, BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model:

        x = Linear(x)
        x = x * scale_factor
        x = x + x
        x = clamp(x, clamp_min, clamp_max)
        x = logsumexp(x, dim=1, keepdim=True)
        x = x * mish(x)

    Implemented as a single fused Triton kernel without materializing the [M, N] intermediate.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Initialize like nn.Linear for similar statistics
        linear = nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())

        # Fused scale factor (x * scale_factor; then x + x -> x * (2 * scale_factor))
        self.scale_total = 2.0 * float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        # Move input to CUDA if necessary
        if not x.is_cuda:
            x = x.cuda()

        x = fused_linear_logsumexp_mish(
            x, self.weight, self.bias,
            self.scale_total, self.clamp_min, self.clamp_max
        )
        return x
