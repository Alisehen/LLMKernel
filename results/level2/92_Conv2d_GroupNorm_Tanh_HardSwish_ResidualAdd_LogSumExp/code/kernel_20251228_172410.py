import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    H, W, H_out, W_out,
    K_total, KH, KW,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    M, HW_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # along flattened output spatial+batch dimension
    pid_co = tl.program_id(1)  # along output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_co = pid_co * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_co = offs_co < C_out

    # Decompose offs_m -> (n, h_out, w_out)
    n = offs_m // HW_out
    rem_m = offs_m % HW_out
    h_out = rem_m // W_out
    w_out = rem_m % W_out

    # Broadcasted base indices
    n_bc = n[:, None]
    h_bc = h_out[:, None]
    w_bc = w_out[:, None]
    co_bc = offs_co[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    HW_k = KH * KW

    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        # Decompose offs_k -> (ci, kh, kw)
        ci = offs_k // HW_k
        rem_k = offs_k % HW_k
        kh = rem_k // KW
        kw = rem_k % KW

        ci_row = ci[None, :]
        kh_row = kh[None, :]
        kw_row = kw[None, :]

        # A: input "im2col" tile, shape [BLOCK_M, BLOCK_K]
        a_ptrs = (
            x_ptr
            + n_bc * stride_x_n
            + ci_row * stride_x_c
            + (h_bc + kh_row) * stride_x_h
            + (w_bc + kw_row) * stride_x_w
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # B: weight tile, shape [BLOCK_K, BLOCK_N]
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

    # Store result
    y_ptrs = (
        y_ptr
        + n_bc * stride_y_n
        + co_bc * stride_y_c
        + h_bc * stride_y_h
        + w_bc * stride_y_w
    )
    mask = mask_m[:, None] & mask_co[None, :]
    tl.store(y_ptrs, acc, mask=mask)


def triton_conv2d_nchw(x, weight, bias, kernel_size):
    # x: (N, C_in, H, W), weight: (C_out, C_in, KH, KW)
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH_w, KW_w = weight.shape
    assert C_in_w == C_in
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    # Ensure consistency with weight shape
    assert KH == KH_w and KW == KW_w

    H_out = H - KH + 1
    W_out = W - KW + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    M = N * H_out * W_out
    HW_out = H_out * W_out
    K_total = C_in * KH * KW

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        H, W, H_out, W_out,
        K_total, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, HW_out,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    return y


@triton.jit
def groupnorm_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, H, W,
    G, group_size, group_elems,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_mean_n, stride_mean_g,
    stride_rstd_n, stride_rstd_g,
    eps,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    n = pid_n
    g = pid_g

    # Accumulators in fp32
    sum_val = 0.0
    sum_sq = 0.0

    HW = H * W

    for off in range(0, group_elems, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < group_elems

        c_in_group = offs // HW
        rem = offs % HW
        h = rem // W
        w = rem % W

        c = g * group_size + c_in_group

        ptrs = (
            x_ptr
            + n * stride_x_n
            + c * stride_x_c
            + h * stride_x_h
            + w * stride_x_w
        )
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    # Convert group_elems to float
    group_elems_f = group_elems * 1.0
    mean = sum_val / group_elems_f
    var = sum_sq / group_elems_f - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    mean_ptr_n = mean_ptr + n * stride_mean_n + g * stride_mean_g
    rstd_ptr_n = rstd_ptr + n * stride_rstd_n + g * stride_rstd_g

    tl.store(mean_ptr_n, mean)
    tl.store(rstd_ptr_n, rstd)


def groupnorm_stats_triton(x, groups, eps):
    N, C, H, W = x.shape
    assert C % groups == 0
    group_size = C // groups
    group_elems = group_size * H * W

    mean = torch.empty((N, groups), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)

    grid = lambda META: (N, groups)

    groupnorm_stats_kernel[grid](
        x, mean, rstd,
        N, C, H, W,
        groups, group_size, group_elems,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        mean.stride(0), mean.stride(1),
        rstd.stride(0), rstd.stride(1),
        eps,
        BLOCK=128,
        num_warps=4,
    )
    return mean, rstd


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
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    n = pid_n
    S = H * W
    s = pid_s

    # Each program_id(1) corresponds to one spatial position
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
        # Conv2d
        x_conv = triton_conv2d_nchw(x, self.weight, self.bias, self.kernel_size)
        # GroupNorm stats
        mean, rstd = groupnorm_stats_triton(x_conv, self.groups, self.eps)
        # GroupNorm + Tanh + HardSwish + Residual + LogSumExp
        x_logsumexp = fused_groupnorm_act_res_lse_triton(
            x_conv, mean, rstd, self.gn_weight, self.gn_bias, self.groups
        )
        return x_logsumexp
