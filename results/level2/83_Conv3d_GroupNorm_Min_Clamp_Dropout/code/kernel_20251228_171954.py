import torch, torch.nn as nn, triton, triton.language as tl


# ---------------------------------------------------------------
# GroupNorm + min + clamp + dropout fused kernel
# ---------------------------------------------------------------

@triton.jit
def groupnorm_min_clamp_dropout_kernel(
    x_ptr,        # *f16/f32, [N, C, D, H, W]
    gamma_ptr,    # *f16/f32, [C]
    beta_ptr,     # *f16/f32, [C]
    y_ptr,        # *f16/f32, [N, C, D, H, W]
    N, C, D, H, W,
    groups,
    channels_per_group,
    group_size,   # channels_per_group * D * H * W
    stride_n, stride_c, stride_d, stride_h, stride_w,
    eps, min_value, max_value,
    p,            # dropout probability
    seed,         # int32
    IS_TRAINING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # We launch exactly N * groups programs, so no bounds check needed.
    total_groups = N * groups

    # Map pid -> (n, g)
    n = pid // groups
    g = pid % groups

    c_start = g * channels_per_group

    # First pass: compute mean and variance over the group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    offs_vec = tl.arange(0, BLOCK_SIZE)

    for offset in range(0, group_size, BLOCK_SIZE):
        idx = offset + offs_vec  # [BLOCK_SIZE]
        mask = idx < group_size

        cur = idx
        w_idx = cur % W
        cur = cur // W
        h_idx = cur % H
        cur = cur // H
        d_idx = cur % D
        cur = cur // D
        c_local = cur
        c = c_start + c_local  # [BLOCK_SIZE]

        x_ptrs = (
            x_ptr
            + n * stride_n
            + c * stride_c
            + d_idx * stride_d
            + h_idx * stride_h
            + w_idx * stride_w
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        sum_val += tl.sum(x_f32, axis=0)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)

    m = tl.full((), group_size, tl.float32)
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, min, clamp, dropout, and store
    for offset in range(0, group_size, BLOCK_SIZE):
        idx = offset + offs_vec
        mask = idx < group_size

        cur = idx
        w_idx = cur % W
        cur = cur // W
        h_idx = cur % H
        cur = cur // H
        d_idx = cur % D
        cur = cur // D
        c_local = cur
        c = c_start + c_local

        x_ptrs = (
            x_ptr
            + n * stride_n
            + c * stride_c
            + d_idx * stride_d
            + h_idx * stride_h
            + w_idx * stride_w
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)

        # GroupNorm
        x_hat = (x_f32 - mean) * rstd
        y = x_hat * gamma + beta

        # torch.min(x, min_value)
        y = tl.minimum(y, min_value)
        # torch.clamp(y, min=min_value, max=max_value)
        y = tl.maximum(y, min_value)
        y = tl.minimum(y, max_value)

        if IS_TRAINING:
            # Dropout: per-element random
            lin = (((n * C + c) * D + d_idx) * H + h_idx) * W + w_idx
            lin = lin.to(tl.int32)
            rnd = tl.rand(seed, lin)
            keep = rnd > p
            scale = 1.0 / (1.0 - p)
            y = tl.where(keep, y * scale, 0.0)

        y_cast = y.to(x.dtype)

        y_ptrs = (
            y_ptr
            + n * stride_n
            + c * stride_c
            + d_idx * stride_d
            + h_idx * stride_h
            + w_idx * stride_w
        )
        tl.store(y_ptrs, y_cast, mask=mask)


def groupnorm_min_clamp_dropout_triton(
    x,
    gamma,
    beta,
    groups,
    eps,
    min_value,
    max_value,
    p,
    training: bool,
):
    """
    x:     [N, C, D, H, W]
    gamma: [C]
    beta:  [C]
    """
    N, C, D, H, W = x.shape
    assert C % groups == 0
    channels_per_group = C // groups
    group_size = channels_per_group * D * H * W
    total_groups = N * groups

    y = torch.empty_like(x)

    strides = x.stride()

    BLOCK_SIZE = 128  # power-of-2

    grid = lambda META: (total_groups,)

    # Simple random seed; not synchronized with PyTorch RNG, but sufficient
    seed = int(torch.randint(0, 2**31 - 1, (1,), device=x.device).item())

    IS_TRAINING = bool(training and (p > 0.0))

    groupnorm_min_clamp_dropout_kernel[grid](
        x, gamma, beta, y,
        N, C, D, H, W,
        groups,
        channels_per_group,
        group_size,
        strides[0], strides[1], strides[2], strides[3], strides[4],
        float(eps), float(min_value), float(max_value),
        float(p),
        seed,
        IS_TRAINING=IS_TRAINING,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---------------------------------------------------------------
# High-level conv3d wrapper using cuDNN (PyTorch) + Triton fusion
# ---------------------------------------------------------------

def conv3d_triton(x, weight, bias):
    """
    Wrapper that delegates 3D convolution to PyTorch/cuDNN for maximum
    performance, matching nn.Conv3d with stride=1, padding=0, dilation=1.
    """
    return torch.nn.functional.conv3d(
        x,
        weight,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )


# ---------------------------------------------------------------
# PyTorch Module using cuDNN conv3d + Triton post-processing
# ---------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Model that performs:
      - 3D convolution (stride=1, padding=0, dilation=1) via cuDNN,
      - Group Normalization,
      - torch.min with scalar,
      - torch.clamp with scalar min/max,
      - Dropout.

    GroupNorm + min + clamp + dropout are fused in a Triton kernel.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        min_value,
        max_value,
        dropout_p,
    ):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            assert len(kernel_size) == 3
            kD, kH, kW = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kD, kH, kW)

        # Conv3d parameters (no padding, stride=1), matching nn.Conv3d defaults
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kD, kH, kW)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # GroupNorm parameters
        assert out_channels % groups == 0
        self.groups = groups
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = 1e-5  # matches nn.GroupNorm default

        # Min / clamp / dropout parameters
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.dropout_p = float(dropout_p)

    def forward(self, x):
        # 3D convolution via cuDNN
        x = conv3d_triton(x, self.weight, self.bias)
        # GroupNorm + min + clamp + dropout fused in Triton
        x = groupnorm_min_clamp_dropout_triton(
            x,
            self.gn_weight,
            self.gn_bias,
            self.groups,
            self.eps,
            self.min_value,
            self.max_value,
            self.dropout_p,
            self.training,
        )
        return x
