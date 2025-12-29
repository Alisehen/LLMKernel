import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_with_stats_kernel(
    x_ptr, w_ptr, b_ptr,
    y_ptr, sum_ptr, sum_sq_ptr,
    N, C_in, C_out,
    H, W, H_out, W_out,
    K_total, KH, KW,
    G, group_size,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    stride_sum_n, stride_sum_g,
    stride_sum_sq_n, stride_sum_sq_g,
    HW_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused Conv2d (NCHW, stride=1, padding=0) + partial GroupNorm stats.

    Grid:
      pid_n  = program_id(0) in [0, N)
      pid_hw = program_id(1) in [0, ceil(H_out*W_out / BLOCK_M))
      pid_co = program_id(2) in [0, ceil(C_out / BLOCK_N))

    Each program computes:
      - A tile of output y[n, co, h, w] for a fixed batch n
      - Partial sums / sums of squares for GroupNorm groups for that batch n,
        accumulated over its (co, h, w) tile and atomically added to global
        per-(n, g) accumulators.
    """
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_co = tl.program_id(2)

    n = pid_n

    # Spatial tile indices (flattened h*w)
    offs_hw = pid_hw * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_hw = offs_hw < HW_out

    h_out = offs_hw // W_out
    w_out = offs_hw % W_out

    # Channel tile indices
    offs_co = pid_co * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_co = offs_co < C_out

    # Broadcasted indices
    h_bc = h_out[:, None]
    w_bc = w_out[:, None]
    co_bc = offs_co[None, :]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    HW_k = KH * KW

    # K loop over (ci, kh, kw)
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        ci = offs_k // HW_k
        rem_k = offs_k % HW_k
        kh = rem_k // KW
        kw = rem_k % KW

        ci_row = ci[None, :]
        kh_row = kh[None, :]
        kw_row = kw[None, :]

        # Input tile A: [BLOCK_M, BLOCK_K]
        a_ptrs = (
            x_ptr
            + n * stride_x_n
            + ci_row * stride_x_c
            + (h_bc + kh_row) * stride_x_h
            + (w_bc + kw_row) * stride_x_w
        )
        a = tl.load(
            a_ptrs,
            mask=mask_hw[:, None] & mask_k[None, :],
            other=0.0,
        )

        # Weight tile B: [BLOCK_K, BLOCK_N]
        ci_col = ci[:, None]
        kh_col = kh[:, None]
        kw_col = kw[:, None]

        b_ptrs = (
            w_ptr
            + co_bc * stride_w_co
            + ci_col * stride_w_ci
            + kh_col * stride_w_kh
            + kw_col * stride_w_kw
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_co[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Mask out invalid (h, w, c) positions so they don't contribute to stats
    mask_full = mask_hw[:, None] & mask_co[None, :]
    acc = tl.where(mask_full, acc, 0.0)

    # Store convolution output
    y_ptrs = (
        y_ptr
        + n * stride_y_n
        + co_bc * stride_y_c
        + h_bc * stride_y_h
        + w_bc * stride_y_w
    )
    tl.store(y_ptrs, acc, mask=mask_full)

    # -------- GroupNorm stats accumulation (per n, per group) --------
    # First reduce over spatial tile (BLOCK_M) to get per-channel partial sums
    sum_c = tl.sum(acc, axis=0)          # [BLOCK_N]
    sum_sq_c = tl.sum(acc * acc, axis=0) # [BLOCK_N]

    # offs_co: [BLOCK_N]
    # For each group g, accumulate contributions from channels within that group
    for g in range(0, G):
        group_co_start = g * group_size
        group_co_end = group_co_start + group_size

        # Bool mask of channels in this tile belonging to group g and valid
        mask_g = (
            (offs_co >= group_co_start)
            & (offs_co < group_co_end)
            & mask_co
        )

        # If no channels of this group in this tile, contribution is zero
        part_sum = tl.sum(tl.where(mask_g, sum_c, 0.0), axis=0)
        part_sum_sq = tl.sum(tl.where(mask_g, sum_sq_c, 0.0), axis=0)

        # Atomic add to global accumulators for (n, g)
        sum_ptr_ng = sum_ptr + n * stride_sum_n + g * stride_sum_g
        sum_sq_ptr_ng = sum_sq_ptr + n * stride_sum_sq_n + g * stride_sum_sq_g

        tl.atomic_add(sum_ptr_ng, part_sum)
        tl.atomic_add(sum_sq_ptr_ng, part_sum_sq)


@triton.jit
def groupnorm_finalize_stats_kernel(
    sum_ptr, sum_sq_ptr, mean_ptr, rstd_ptr,
    N, G, group_elems,
    stride_sum_n, stride_sum_g,
    stride_sum_sq_n, stride_sum_sq_g,
    stride_mean_n, stride_mean_g,
    stride_rstd_n, stride_rstd_g,
    eps,
    BLOCK_G: tl.constexpr,
):
    """
    Convert accumulated per-(n, g) sums and sums of squares into mean and rstd:
      mean = sum / group_elems
      var  = sum_sq / group_elems - mean^2
      rstd = 1 / sqrt(var + eps)
    """
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    n = pid_n
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < G

    # Load sums
    sum_ptrs = sum_ptr + n * stride_sum_n + offs_g * stride_sum_g
    sum_sq_ptrs = sum_sq_ptr + n * stride_sum_sq_n + offs_g * stride_sum_sq_g

    sum_val = tl.load(sum_ptrs, mask=mask_g, other=0.0)
    sum_sq_val = tl.load(sum_sq_ptrs, mask=mask_g, other=0.0)

    group_elems_f = group_elems * 1.0
    mean = sum_val / group_elems_f
    var = sum_sq_val / group_elems_f - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    mean_ptrs = mean_ptr + n * stride_mean_n + offs_g * stride_mean_g
    rstd_ptrs = rstd_ptr + n * stride_rstd_n + offs_g * stride_rstd_g

    tl.store(mean_ptrs, mean, mask=mask_g)
    tl.store(rstd_ptrs, rstd, mask=mask_g)


def triton_conv2d_nchw_with_gn_stats(x, weight, bias, groups, eps):
    """
    Fused conv2d (NCHW, stride=1, padding=0) + GroupNorm stats computation.

    Returns:
      y     : conv output, shape (N, C_out, H_out, W_out)
      mean  : per-(N, group) mean, shape (N, groups)
      rstd  : per-(N, group) rstd, shape (N, groups)
    """
    # x: (N, C_in, H, W), weight: (C_out, C_in, KH, KW)
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH_w, KW_w = weight.shape
    assert C_in_w == C_in

    if isinstance(weight, torch.nn.Parameter):
        weight_tensor = weight
    else:
        weight_tensor = weight

    # Infer kernel size
    KH = KH_w
    KW = KW_w

    H_out = H - KH + 1
    W_out = W - KW + 1
    HW_out = H_out * W_out

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    assert C_out % groups == 0
    group_size = C_out // groups
    G = groups
    group_elems = group_size * H_out * W_out

    # Per-(N, G) accumulators for sum and sum of squares (fp32)
    sum = torch.zeros((N, G), device=x.device, dtype=torch.float32)
    sum_sq = torch.zeros_like(sum)

    K_total = C_in * KH * KW

    grid = lambda META: (
        N,
        triton.cdiv(HW_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_with_stats_kernel[grid](
        x, weight_tensor, bias,
        y, sum, sum_sq,
        N, C_in, C_out,
        H, W, H_out, W_out,
        K_total, KH, KW,
        G, group_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight_tensor.stride(0), weight_tensor.stride(1),
        weight_tensor.stride(2), weight_tensor.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        sum.stride(0), sum.stride(1),
        sum_sq.stride(0), sum_sq.stride(1),
        HW_out,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )

    # Finalize GroupNorm statistics from accumulators
    mean = torch.empty_like(sum)
    rstd = torch.empty_like(sum)

    grid_stats = lambda META: (
        N,
        triton.cdiv(G, META["BLOCK_G"]),
    )

    groupnorm_finalize_stats_kernel[grid_stats](
        sum, sum_sq, mean, rstd,
        N, G, group_elems,
        sum.stride(0), sum.stride(1),
        sum_sq.stride(0), sum_sq.stride(1),
        mean.stride(0), mean.stride(1),
        rstd.stride(0), rstd.stride(1),
        eps,
        BLOCK_G=32,
        num_warps=4,
    )

    return y, mean, rstd


@triton.jit
def fused_groupnorm_act_res_lse_kernel(
    x_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, H, W,
    G, group_size,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_mean_n, stride_mean_g,
    stride_rstd_n, stride_rstd_g,
    stride_gamma_c, stride_beta_c,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel:
      GroupNorm -> Tanh -> HardSwish -> Residual Add -> LogSumExp over C.

    x_ptr:    conv output, shape (N, C, H, W)
    mean_ptr: (N, G)
    rstd_ptr: (N, G)
    gamma,beta: GroupNorm affine params, length C
    out_ptr:  (N, 1, H, W)
    """
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    n = pid_n
    S = H * W
    s = pid_s

    h = s // W
    w = s % W

    # First pass: compute max over channels of residual x_res
    max_val = -float("inf")

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_ptrs = (
            x_ptr
            + n * stride_x_n
            + offs_c * stride_x_c
            + h * stride_x_h
            + w * stride_x_w
        )
        x = tl.load(x_ptrs, mask=mask_c, other=0.0)

        g = offs_c // group_size

        mean_ptrs = mean_ptr + n * stride_mean_n + g * stride_mean_g
        rstd_ptrs = rstd_ptr + n * stride_rstd_n + g * stride_rstd_g
        mean = tl.load(mean_ptrs, mask=mask_c, other=0.0)
        rstd = tl.load(rstd_ptrs, mask=mask_c, other=0.0)

        gamma_ptrs = gamma_ptr + offs_c * stride_gamma_c
        beta_ptrs = beta_ptr + offs_c * stride_beta_c
        gamma = tl.load(gamma_ptrs, mask=mask_c, other=0.0)
        beta = tl.load(beta_ptrs, mask=mask_c, other=0.0)

        x_hat = (x - mean) * rstd
        x_aff = x_hat * gamma + beta

        # tanh(x_aff) via exp(2x)
        two_x = 2.0 * x_aff
        exp2x = tl.exp(two_x)
        tanh_x = (exp2x - 1.0) / (exp2x + 1.0)

        # HardSwish(tanh_x)
        t = tanh_x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        x_hs = tanh_x * t * (1.0 / 6.0)

        x_res = x + x_hs
        x_res = tl.where(mask_c, x_res, -float("inf"))

        block_max = tl.max(x_res, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute log-sum-exp using max_val
    sum_exp = 0.0

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_ptrs = (
            x_ptr
            + n * stride_x_n
            + offs_c * stride_x_c
            + h * stride_x_h
            + w * stride_x_w
        )
        x = tl.load(x_ptrs, mask=mask_c, other=0.0)

        g = offs_c // group_size

        mean_ptrs = mean_ptr + n * stride_mean_n + g * stride_mean_g
        rstd_ptrs = rstd_ptr + n * stride_rstd_n + g * stride_rstd_g
        mean = tl.load(mean_ptrs, mask=mask_c, other=0.0)
        rstd = tl.load(rstd_ptrs, mask=mask_c, other=0.0)

        gamma_ptrs = gamma_ptr + offs_c * stride_gamma_c
        beta_ptrs = beta_ptr + offs_c * stride_beta_c
        gamma = tl.load(gamma_ptrs, mask=mask_c, other=0.0)
        beta = tl.load(beta_ptrs, mask=mask_c, other=0.0)

        x_hat = (x - mean) * rstd
        x_aff = x_hat * gamma + beta

        # tanh(x_aff)
        two_x = 2.0 * x_aff
        exp2x = tl.exp(two_x)
        tanh_x = (exp2x - 1.0) / (exp2x + 1.0)

        # HardSwish(tanh_x)
        t = tanh_x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        x_hs = tanh_x * t * (1.0 / 6.0)

        x_res = x + x_hs
        x_res = tl.where(mask_c, x_res, -float("inf"))

        sum_exp += tl.sum(tl.exp(x_res - max_val), axis=0)

    lse = tl.log(sum_exp) + max_val

    out_ptrs = (
        out_ptr
        + n * stride_out_n
        + 0 * stride_out_c
        + h * stride_out_h
        + w * stride_out_w
    )
    tl.store(out_ptrs, lse)


def fused_groupnorm_act_res_lse_triton(x, mean, rstd, gamma, beta, groups):
    """
    Wrapper for fused GroupNorm + activation + residual + LogSumExp kernel.

    x:     (N, C, H, W)
    mean:  (N, groups)
    rstd:  (N, groups)
    gamma: (C,)
    beta:  (C,)
    """
    N, C, H, W = x.shape
    group_size = C // groups

    out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (N, H * W)

    fused_groupnorm_act_res_lse_kernel[grid](
        x, mean, rstd, gamma, beta, out,
        N, C, H, W,
        groups, group_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        mean.stride(0), mean.stride(1),
        rstd.stride(0), rstd.stride(1),
        gamma.stride(0), beta.stride(0),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=64,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the model:
    Conv2d -> GroupNorm -> Tanh -> HardSwish -> Residual Add -> LogSumExp

    Optimization:
      - Fuses Conv2d with GroupNorm statistics computation to avoid an extra
        full pass over the conv output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # Conv weights and bias
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

        # GroupNorm affine parameters
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Conv2d + GroupNorm stats (fused)
        x_conv, mean, rstd = triton_conv2d_nchw_with_gn_stats(
            x, self.weight, self.bias, self.groups, self.eps
        )
        # GroupNorm + Tanh + HardSwish + Residual + LogSumExp
        x_logsumexp = fused_groupnorm_act_res_lse_triton(
            x_conv, mean, rstd, self.gn_weight, self.gn_bias, self.groups
        )
        return x_logsumexp
