import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def rowwise_scale_clamp_logsumexp_mish_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om,
    total_scale, clamp_min, clamp_max,
    BLOCK_N: tl.constexpr,
):
    """
    x_ptr:   (M, N) linear output (matmul + bias), row-major
    out_ptr: (M, 1) result: logsumexp over N, then x * mish(x)
    """
    pid_m = tl.program_id(0)
    # Each program processes exactly one row: pid_m in [0, M)
    # Grid is launched with size = M, so no bounds check is needed.

    neg_inf = -1.0e30
    row_max = tl.full((), neg_inf, dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # Base pointer for this row
    row_x_ptr = x_ptr + pid_m * stride_xm

    # Streaming logsumexp over N with fused scale + clamp
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        x = tl.load(row_x_ptr + offs_n * stride_xn, mask=mask_n, other=0.0)
        x = x.to(tl.float32)

        # Fused scaling (includes residual x + x -> factor 2) and clamp
        y = x * total_scale
        y = tl.maximum(y, clamp_min)
        y = tl.minimum(y, clamp_max)

        # Mask out invalid columns so they don't affect max/sum
        y = tl.where(mask_n, y, neg_inf)

        # Tile-wise max
        tile_max = tl.max(y, axis=0)

        # Streaming logsumexp update
        m_old = row_max
        s_old = row_sum
        m_new = tl.maximum(m_old, tile_max)

        y_shift = y - m_new
        exp_tile = tl.exp(y_shift)
        sum_tile = tl.sum(exp_tile, axis=0)

        row_sum = s_old * tl.exp(m_old - m_new) + sum_tile
        row_max = m_new

    # Final logsumexp for this row
    lse = tl.log(row_sum) + row_max

    # Mish activation on the scalar lse:
    # mish(x) = x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(lse))
    t = tl.exp(2.0 * sp)
    tanh_sp = (t - 1.0) / (t + 1.0)
    mish_lse = lse * tanh_sp

    out_val = lse * mish_lse  # x * mish(x)

    # Store result with keepdim=True shape (M, 1)
    out_ptrs = out_ptr + pid_m * stride_om
    tl.store(out_ptrs, out_val)


def linear_scale_clamp_logsumexp_mish(x, weight, bias, scale_factor, clamp_min, clamp_max):
    """
    x:       (M, K)
    weight:  (N, K)  -- nn.Linear(out_features=N, in_features=K).weight
    bias:    (N,)
    returns: (M, 1)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Phase 1: use cuBLAS-backed GEMM via F.linear for matmul + bias
    # y: (M, N) = x @ weight.T + bias
    y = F.linear(x, weight, bias)

    # Output: logsumexp over dim=1 with keepdim=True
    out = torch.empty((M, 1), device=x.device, dtype=y.dtype)

    # Fold "x * scale_factor; x = x + x" into a single multiplier
    total_scale = float(scale_factor) * 2.0

    grid = lambda META: (triton.cdiv(M, 1),)

    rowwise_scale_clamp_logsumexp_mish_kernel[grid](
        y, out,
        M, N,
        y.stride(0), y.stride(1),
        out.stride(0),
        total_scale, clamp_min, clamp_max,
        BLOCK_N=256,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model.

    Phase 1: Linear (matmul + bias) via cuBLAS-backed torch.nn.functional.linear
    Phase 2: Fused scale + residual-add (as a single multiply), clamp, row-wise
             logsumexp, Mish, and final x * mish(x) via Triton reduction kernel.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # x: (batch_size, input_size)
        return linear_scale_clamp_logsumexp_mish(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )
