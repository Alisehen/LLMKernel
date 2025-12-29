# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, Cin, H, W, Cout, Ho, Wo, Kh, Kw, Ktotal, P,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < Cout

    # Map flattened M index -> (b, ho, wo)
    HoWo = Ho * Wo
    offs_m_b = offs_m // HoWo
    rem_m = offs_m - offs_m_b * HoWo
    offs_m_ho = rem_m // Wo
    offs_m_wo = rem_m - offs_m_ho * Wo

    # Make them 2D [BM,1]
    b = offs_m_b[:, None]
    ho = offs_m_ho[:, None]
    wo = offs_m_wo[:, None]

    # N indices [1, BN]
    n = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, Ktotal, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Ktotal

        # Map K index -> (cin, kh, kw)
        KhKw = Kh * Kw
        cin = offs_k // KhKw
        rem_k = offs_k - cin * KhKw
        kh = rem_k // Kw
        kw = rem_k - kh * Kw

        # shapes: cin,kh,kw -> [BK]; broadcast later
        cin_b = cin[None, :]       # [1, BK]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Input pointers [BM, BK]
        x_ptrs = (
            x_ptr
            + b * stride_xb
            + cin_b * stride_xc
            + (ho + kh_b) * stride_xh
            + (wo + kw_b) * stride_xw
        )

        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # Weight pointers [BK, BN]
        w_ptrs = (
            w_ptr
            + cin[:, None] * stride_wci
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
            + n * stride_wco
        )

        mask_b = mask_k[:, None] & mask_n[None, :]
        b_mat = tl.load(w_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b_mat, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + b * stride_yb
        + n * stride_yc
        + ho * stride_yh
        + wo * stride_yw
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def conv2d_triton(x, weight, bias):
    """
    x: (B, Cin, H, W)
    weight: (Cout, Cin, Kh, Kw)
    bias: (Cout,)
    Returns: (B, Cout, Ho, Wo) with stride=1, padding=0
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, Cin, H, W = x.shape
    Cout, Cin_w, Kh, Kw = weight.shape
    assert Cin == Cin_w
    Ho = H - Kh + 1
    Wo = W - Kw + 1
    Ktotal = Cin * Kh * Kw
    P = B * Ho * Wo

    y = torch.empty((B, Cout, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(P, META["BLOCK_M"]),
        triton.cdiv(Cout, META["BLOCK_N"]),
    )

    conv2d_forward_kernel[grid](
        x, weight, bias, y,
        B, Cin, H, W, Cout, Ho, Wo, Kh, Kw, Ktotal, P,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return y


@triton.jit
def groupnorm_stats_kernel(
    x_ptr, mean_ptr, var_ptr,
    B, C, H, W, G, channels_per_group, M,
    stride_xb, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. B*G-1
    b = pid // G
    g = pid - b * G

    # total elements per group = M = channels_per_group * H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    HW = H * W

    for offset in range(0, M, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < M

        ch_idx = offs // HW
        sp_idx = offs - ch_idx * HW
        h_idx = sp_idx // W
        w_idx = sp_idx - h_idx * W

        c = g * channels_per_group + ch_idx

        x_ptrs = (
            x_ptr
            + b * stride_xb
            + c * stride_xc
            + h_idx * stride_xh
            + w_idx * stride_xw
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)

    M_f = tl.float32(M)
    mean = sum_val / M_f
    var = sum_sq_val / M_f - mean * mean

    tl.store(mean_ptr + pid, mean)
    tl.store(var_ptr + pid, var)


def groupnorm_stats_triton(x, num_groups):
    """
    x: (B, C, H, W)
    Returns:
        mean: (B*num_groups,) float32
        var:  (B*num_groups,) float32
    """
    assert x.is_cuda
    B, C, H, W = x.shape
    G = num_groups
    assert C % G == 0
    channels_per_group = C // G
    M = channels_per_group * H * W

    mean = torch.empty(B * G, device=x.device, dtype=torch.float32)
    var = torch.empty_like(mean)

    BLOCK_SIZE = 256
    grid = lambda META: (B * G,)

    groupnorm_stats_kernel[grid](
        x, mean, var,
        B, C, H, W, G, channels_per_group, M,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return mean, var


@triton.jit
def groupnorm_scale_maxpool_clamp_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, scale_ptr, y_ptr,
    B, C, H, W, G, channels_per_group,
    Hpo, Wpo, Kp, stride_p, Nout,
    eps, clamp_min, clamp_max,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < Nout

    HWpo = Hpo * Wpo
    per_batch = C * HWpo

    b = offs // per_batch
    rem1 = offs - b * per_batch
    c = rem1 // HWpo
    rem2 = rem1 - c * HWpo
    ho_p = rem2 // Wpo
    wo_p = rem2 - ho_p * Wpo

    group = c // channels_per_group
    mean_idx = b * G + group

    mean = tl.load(mean_ptr + mean_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + mean_idx, mask=mask, other=0.0)

    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(weight_ptr + c, mask=mask, other=1.0)
    beta = tl.load(bias_ptr + c, mask=mask, other=0.0)
    scale_v = tl.load(scale_ptr + c, mask=mask, other=1.0)

    max_val = tl.zeros((BLOCK,), dtype=tl.float32) + (-1.0e30)

    for kh in range(0, Kp):
        for kw in range(0, Kp):
            h_in = ho_p * stride_p + kh
            w_in = wo_p * stride_p + kw

            x_ptrs = (
                x_ptr
                + b * stride_xb
                + c * stride_xc
                + h_in * stride_xh
                + w_in * stride_xw
            )

            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

            xn = (x - mean) * rstd
            xn = xn * gamma + beta
            xn = xn * scale_v

            max_val = tl.maximum(max_val, xn)

    max_val = tl.maximum(max_val, clamp_min)
    max_val = tl.minimum(max_val, clamp_max)

    y_ptrs = (
        y_ptr
        + b * stride_yb
        + c * stride_yc
        + ho_p * stride_yh
        + wo_p * stride_yw
    )
    tl.store(y_ptrs, max_val, mask=mask)


def groupnorm_scale_maxpool_clamp_triton(
    x, mean, var, weight, bias, scale,
    num_groups, pool_ks, clamp_min, clamp_max, eps,
):
    """
    x: (B, C, H, W) conv output
    mean, var: (B*num_groups,) from groupnorm_stats_triton
    weight, bias: GroupNorm affine params, shape (C,)
    scale: (C, 1, 1) or (C,)
    """
    assert x.is_cuda
    B, C, H, W = x.shape
    G = num_groups
    assert C % G == 0
    channels_per_group = C // G

    stride_p = pool_ks
    Hpo = (H - pool_ks) // stride_p + 1
    Wpo = (W - pool_ks) // stride_p + 1
    Nout = B * C * Hpo * Wpo

    y = torch.empty((B, C, Hpo, Wpo), device=x.device, dtype=x.dtype)

    # Ensure 1D views for these parameters
    mean_flat = mean
    var_flat = var
    weight_flat = weight.view(-1)
    bias_flat = bias.view(-1)
    scale_flat = scale.view(-1)

    BLOCK = 256
    grid = lambda META: (triton.cdiv(Nout, META["BLOCK"]),)

    groupnorm_scale_maxpool_clamp_kernel[grid](
        x, mean_flat, var_flat, weight_flat, bias_flat, scale_flat, y,
        B, C, H, W, G, channels_per_group,
        Hpo, Wpo, pool_ks, stride_p, Nout,
        eps, clamp_min, clamp_max,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=BLOCK,
        num_warps=4, num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model:
    Conv2d -> GroupNorm -> scale -> MaxPool2d -> clamp
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 num_groups, scale_shape, maxpool_kernel_size,
                 clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Conv via Triton
        x = conv2d_triton(x, self.conv.weight, self.conv.bias)

        # GroupNorm statistics via Triton
        mean, var = groupnorm_stats_triton(x, self.group_norm.num_groups)

        # GroupNorm + scale + maxpool + clamp fused via Triton
        x = groupnorm_scale_maxpool_clamp_triton(
            x,
            mean,
            var,
            self.group_norm.weight,
            self.group_norm.bias,
            self.scale,
            self.group_norm.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max,
            self.group_norm.eps,
        )
        return x
