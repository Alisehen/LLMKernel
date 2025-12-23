import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_scale_clamp_logsumexp_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om,
    scale, clamp_min, clamp_max,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id along batch (M) dimension
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Row-wise running max and sum for streaming logsumexp
    neg_inf = -1.0e30
    row_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over N dimension in tiles
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # GEMM tile: [BLOCK_M, BLOCK_N]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # w_ptr is [N, K] (row-major), but we need [K, N] for dot -> transpose on-the-fly
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        for k in range(0, K, BLOCK_K):
            k_remaining = K - k
            k_mask = offs_k < k_remaining

            x = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptrs,
                mask=k_mask[:, None] & mask_n[None, :],
                other=0.0,
            )

            acc += tl.dot(x, w, allow_tf32=True)

            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # Add bias
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias[None, :]

        # Fused scaling (includes residual x + x -> factor 2) and clamp
        acc = acc * scale
        acc = tl.maximum(acc, clamp_min)
        acc = tl.minimum(acc, clamp_max)

        # Mask out invalid columns before reduction so they don't affect max/sum
        valid_acc = tl.where(mask_n[None, :], acc, neg_inf)

        # Tile-wise row max
        tile_max = tl.max(valid_acc, axis=1)
        tile_max = tl.where(mask_m, tile_max, row_max)

        # Streaming logsumexp update:
        # m_new = max(m_old, tile_max)
        # s_new = s_old * exp(m_old - m_new) + sum(exp(tile - m_new))
        m_old = row_max
        s_old = row_sum
        m_new = tl.maximum(m_old, tile_max)

        acc_shift = valid_acc - m_new[:, None]
        exp_tile = tl.exp(acc_shift)
        sum_tile = tl.sum(exp_tile, axis=1)

        row_sum = s_old * tl.exp(m_old - m_new) + sum_tile
        row_max = m_new

    # Final logsumexp per row: logsumexp = log(row_sum) + row_max
    lse = tl.log(row_sum) + row_max

    # Mish activation: mish(x) = x * tanh(softplus(x)),
    # softplus(x) = log(1 + exp(x)), tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
    sp = tl.log(1.0 + tl.exp(lse))
    t = tl.exp(2.0 * sp)
    tanh_sp = (t - 1.0) / (t + 1.0)
    mish_lse = lse * tanh_sp

    out_val = lse * mish_lse  # x * mish(x)

    # Store result with keepdim=True shape (M, 1)
    out_ptrs = out_ptr + offs_m * stride_om
    tl.store(out_ptrs, out_val, mask=mask_m)


def fused_linear_scale_clamp_logsumexp_mish(x, weight, bias, scale_factor, clamp_min, clamp_max):
    """
    x:       (M, K)
    weight:  (N, K)  -- same as nn.Linear(out_features=N, in_features=K).weight
    bias:    (N,)
    returns: (M, 1)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Output: logsumexp over dim=1 with keepdim=True
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    # Fold "x * scale_factor; x = x + x" into a single constant multiplier
    total_scale = float(scale_factor) * 2.0

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    fused_linear_scale_clamp_logsumexp_mish_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0),
        total_scale, clamp_min, clamp_max,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model.
    Fuses: Linear (matmul + bias) + scale + residual add + clamp + logsumexp + Mish + multiply.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Keep original structure / initialization via nn.Linear
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # x: (batch_size, input_size)
        # Use Triton kernel with the linear layer's parameters
        return fused_linear_scale_clamp_logsumexp_mish(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )
