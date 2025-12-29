import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W,
    C_out, Kh, Kw,
    H_out, W_out, M, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs along M (rows = N * H_out * W_out) and N (cols = C_out)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in output matrix (M x C_out)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    HW_out = H_out * W_out

    # Map row index -> (n, oh, ow)
    n_idx = offs_m // HW_out
    hw = offs_m % HW_out
    oh = hw // W_out
    ow = hw % W_out

    # Accumulator for a BLOCK_M x BLOCK_N tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = C_in * Kh * Kw, in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decompose K index into (ci, kh, kw)
        ci = offs_k // (Kh * Kw)
        kk2 = offs_k % (Kh * Kw)
        kh = kk2 // Kw
        kw = kk2 % Kw

        # A tile: input patches, shape [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + (oh[:, None] + kh[None, :]) * stride_xh
            + (ow[:, None] + kw[None, :]) * stride_xw
            + ci[None, :] * stride_xc
        )
        x_vals = tl.load(
            x_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # B tile: weights reshaped as [K, C_out], load [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wn
            + ci[:, None] * stride_wc
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
        )
        w_vals = tl.load(
            w_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # GEMM-style accumulate
        acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store result back to NCHW output tensor
    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_on
        + offs_n[None, :] * stride_oc
        + oh[:, None] * stride_oh
        + ow[:, None] * stride_ow
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_triton(x, weight, bias, kernel_size):
    assert x.dim() == 4  # NCHW
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    Kh = Kw = kernel_size
    # Conv2d defaults: stride=1, padding=0, dilation=1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    M = N * H_out * W_out
    K = C_in * Kh * Kw

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out, Kh, Kw,
        H_out, W_out, M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return out


@triton.jit
def groupnorm_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    N, C, H, W,
    groups, C_per_group, L, inv_L,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_mn, stride_mg,
    stride_vn, stride_vg,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_idx = pid // groups
    g_idx = pid % groups
    cg_start = g_idx * C_per_group

    sum_x = 0.0
    sum_x2 = 0.0

    HW = H * W

    for off in range(0, L, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < L

        c_offset = offs // HW
        hw = offs % HW
        h_idx = hw // W
        w_idx = hw % W

        c_idx = cg_start + c_offset

        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + c_idx * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x * inv_L
    mean_x2 = sum_x2 * inv_L
    var = mean_x2 - mean * mean

    mean_ptrs = mean_ptr + n_idx * stride_mn + g_idx * stride_mg
    var_ptrs = var_ptr + n_idx * stride_vn + g_idx * stride_vg
    tl.store(mean_ptrs, mean)
    tl.store(var_ptrs, var)


@triton.jit
def groupnorm_apply_tanh_hswish_residual_kernel(
    x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, H, W,
    groups, C_per_group, eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_mn, stride_mg,
    stride_vn, stride_vg,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    HW = H * W
    n_idx = pid // HW
    hw = pid % HW
    h_idx = hw // W
    w_idx = hw % W

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        g_idx = offs_c // C_per_group

        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + offs_c * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask_c, other=0.0)

        mean_ptrs = mean_ptr + n_idx * stride_mn + g_idx * stride_mg
        var_ptrs = var_ptr + n_idx * stride_vn + g_idx * stride_vg
        mean = tl.load(mean_ptrs, mask=mask_c, other=0.0)
        var = tl.load(var_ptrs, mask=mask_c, other=0.0)

        inv_std = 1.0 / tl.sqrt(var + eps)
        x_norm = (x - mean) * inv_std

        gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0)
        beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)
        y = x_norm * gamma + beta

        # Tanh
        e2 = tl.exp(2.0 * y)
        tanh_y = (e2 - 1.0) / (e2 + 1.0)

        # HardSwish
        hs_input = tanh_y
        hs_inner = hs_input + 3.0
        hs_clamp = tl.minimum(tl.maximum(hs_inner, 0.0), 6.0)
        hs = hs_input * hs_clamp / 6.0

        # Residual addition: x_conv + hswish(tanh(norm(x_conv)))
        res = x + hs

        out_ptrs = (
            out_ptr
            + n_idx * stride_on
            + offs_c * stride_oc
            + h_idx * stride_oh
            + w_idx * stride_ow
        )
        tl.store(out_ptrs, res, mask=mask_c)


def groupnorm_tanh_hswish_residual_triton(x, gamma, beta, groups, eps):
    x = x.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()

    N, C, H, W = x.shape
    assert C % groups == 0
    C_per_group = C // groups

    mean = torch.empty((N, groups), device=x.device, dtype=torch.float32)
    var = torch.empty_like(mean)

    L = C_per_group * H * W
    inv_L = 1.0 / float(L)

    grid_stats = (N * groups,)
    groupnorm_stats_kernel[grid_stats](
        x, mean, var,
        N, C, H, W,
        groups, C_per_group, L, inv_L,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        mean.stride(0), mean.stride(1),
        var.stride(0), var.stride(1),
        BLOCK=64,
    )

    out = torch.empty_like(x)
    grid_apply = (N * H * W,)
    groupnorm_apply_tanh_hswish_residual_kernel[grid_apply](
        x, mean, var, gamma, beta, out,
        N, C, H, W,
        groups, C_per_group, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        mean.stride(0), mean.stride(1),
        var.stride(0), var.stride(1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=64,
    )
    return out


@triton.jit
def logsumexp_channel_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)

    HW = H * W
    n_idx = pid // HW
    hw = pid % HW
    h_idx = hw // W
    w_idx = hw % W

    running_max = -float("inf")
    running_sum = 0.0

    for c_start in range(0, C, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_ptrs = (
            x_ptr
            + n_idx * stride_xn
            + offs_c * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=mask_c, other=-float("inf"))

        block_max = tl.max(x_vals, axis=0)
        new_max = tl.maximum(running_max, block_max)

        exp_old = tl.exp(running_max - new_max) * running_sum
        exp_new = tl.sum(tl.exp(x_vals - new_max), axis=0)
        running_sum = exp_old + exp_new
        running_max = new_max

    result = running_max + tl.log(running_sum)

    out_ptr_scalar = (
        out_ptr
        + n_idx * stride_on
        + 0 * stride_oc
        + h_idx * stride_oh
        + w_idx * stride_ow
    )
    tl.store(out_ptr_scalar, result)


def logsumexp_channel_triton(x):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)

    M = N * H * W
    grid = (M,)

    logsumexp_channel_kernel[grid](
        x, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=64,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version:
      - Conv2d via custom Triton implicit-GEMM kernel
      - GroupNorm + Tanh + HardSwish + Residual in Triton
      - LogSumExp over channels in Triton
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # Conv2d parameters (stride=1, padding=0 to match default nn.Conv2d)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

        # GroupNorm affine parameters (gamma, beta)
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        x_conv = conv2d_triton(x, self.weight, self.bias, self.kernel_size)
        x_res = groupnorm_tanh_hswish_residual_triton(
            x_conv, self.gn_weight, self.gn_bias, self.groups, self.eps
        )
        x_logsumexp = logsumexp_channel_triton(x_res)
        return x_logsumexp
