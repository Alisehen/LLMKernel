import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Matmul + scale + residual + clamp
# -----------------------------

@triton.autotune(
    configs=[
        # Conservative baseline, per instructions
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_scale_residual_clamp_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    fused_scale, clamp_min, clamp_max,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # 1D launch grid with grouped tiling over M for better L2 reuse
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size_m = GROUP_M
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    group_pid = pid % num_pid_in_group

    pid_m = group_id * group_size_m + (group_pid % group_size_m)
    pid_n = group_pid // group_size_m

    # Offsets for this program's tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for M and N dimensions
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to the first K-block for A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM main loop
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k

        a_mask = mask_m[:, None] & (k[None, :] < K)
        b_mask = (k[:, None] < K) & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Bias add (broadcast along M)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Pre-fused: scale_factor * 2 (residual) done on host, just multiply once here
    acc = acc * fused_scale

    # Clamp
    acc = tl.maximum(acc, clamp_min)
    acc = tl.minimum(acc, clamp_max)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_matmul_scale_residual_clamp(x, weight, bias, scale_factor, clamp_min, clamp_max):
    """
    x:      (M, K), float32
    weight: (N, K) as in nn.Linear (out_features, in_features)
    bias:   (N,)
    Returns: (M, N) float32
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Ensure B is laid out as (K, N) contiguous
    b = weight.t().contiguous()  # (K, N)

    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Pre-fuse residual doubling with scale_factor on host to save work in kernel
    fused_scale = float(scale_factor) * 2.0

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) *
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_scale_residual_clamp_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        fused_scale, float(clamp_min), float(clamp_max),
        # meta-parameters filled by autotune
    )

    return c


# -----------------------------
# Row-wise logsumexp + Mish, streaming 1-pass reduction
# -----------------------------

@triton.autotune(
    configs=[
        # Baseline, per instructions
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 512},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N'],
)
@triton.jit
def row_logsumexp_mish_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Tile of rows
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_n = tl.arange(0, BLOCK_N)

    # Initialize streaming logsumexp state per row
    neg_inf = -float("inf")
    max_vals = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # One-pass, numerically stable logsumexp along N
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + offs_n
        mask_n = cols < N
        mask = mask_m[:, None] & mask_n[None, :]

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + cols[None, :] * stride_xn,
            mask=mask,
            other=neg_inf,
        )

        # Current block row-wise max
        block_max = tl.max(x, axis=1)

        # New running max
        new_max = tl.maximum(max_vals, block_max)

        # Rescale old sum_exp to new max
        scale_old = tl.exp(max_vals - new_max)

        # Contribution from current block
        exp_x = tl.exp(x - new_max[:, None])
        block_sum = tl.sum(exp_x, axis=1)

        # Update running sum_exp and max_vals
        sum_exp = sum_exp * scale_old + block_sum
        max_vals = new_max

    # LogSumExp per row
    y = max_vals + tl.log(sum_exp)  # y = lse

    # Optimized Mish implementation:
    # softplus(y) = log(1 + exp(y))
    # tanh(softplus(y)) using u = 1 + exp(y):
    #   exp(softplus) = u
    #   tanh(softplus) = (u^2 - 1) / (u^2 + 1)
    e = tl.exp(y)
    u = 1.0 + e
    softplus = tl.log(u)
    t2 = u * u
    tanh_sp = (t2 - 1.0) / (t2 + 1.0)

    mish_y = y * tanh_sp
    out_val = y * mish_y

    # Store one scalar per row
    tl.store(out_ptr + offs_m * stride_outm, out_val, mask=mask_m)


def fused_row_logsumexp_mish(x):
    """
    x: (M, N) -> out: (M, 1)
    Applies logsumexp over dim=1, then y * mish(y).
    """
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)

    row_logsumexp_mish_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0),
        # meta-parameters filled by autotune
    )

    return out


# -----------------------------
# PyTorch module
# -----------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        y = fused_matmul_scale_residual_clamp(
            x,
            self.matmul.weight,
            self.matmul.bias,
            self.scale_factor,
            self.clamp_min,
            self.clamp_max,
        )
        out = fused_row_logsumexp_mish(y)
        return out
