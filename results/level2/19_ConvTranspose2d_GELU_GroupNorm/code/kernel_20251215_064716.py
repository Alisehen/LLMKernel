# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr,          # *f32 or *f16: [N, C, H, W]
    weight_ptr,     # *f32 or *f16: [C]
    bias_ptr,       # *f32 or *f16: [C]
    N, C, H, W,
    NUM_GROUPS,     # int
    eps,            # float
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU + GroupNorm over [C, H, W] per sample, per group.

    GroupNorm: num_groups = NUM_GROUPS, num_channels = C
    Per group (C_per_group = C / NUM_GROUPS), compute mean/var over
    (C_per_group, H, W) for each N, then apply affine (weight, bias).
    GELU is applied before GroupNorm.
    """
    pid = tl.program_id(axis=0)
    groups = NUM_GROUPS

    # Map program id to (n, g)
    n = pid // groups
    g = pid % groups

    if n >= N:
        return

    channels_per_group = C // groups
    c_start = g * channels_per_group

    # Total elements in this (n, g) group
    group_elems = channels_per_group * H * W

    # Base pointer for this (n, g) slice: (n, c_start, 0, 0)
    base_ptr = x_ptr + n * stride_n + c_start * stride_c

    # First pass: compute mean and variance of GELU(x) over the group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865476

    # Reduction loop
    for offset in range(0, group_elems, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_elems

        # Decode offs -> (c_within_group, h, w)
        c_idx = offs // (H * W)
        hw = offs % (H * W)
        h_idx = hw // W
        w_idx = hw % W

        ptrs = base_ptr + c_idx * stride_c + h_idx * stride_h + w_idx * stride_w
        x = tl.load(ptrs, mask=mask, other=0.0)

        # Promote to f32 for numerics
        x_f32 = x.to(tl.float32)

        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_x = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * inv_sqrt2))

        sum_val += tl.sum(gelu_x, axis=0)
        sum_sq_val += tl.sum(gelu_x * gelu_x, axis=0)

    # Properly create an f32 scalar for the group size (avoid treating dtype as callable)
    group_elems_f32 = tl.full((), group_elems, dtype=tl.float32)

    mean = sum_val / group_elems_f32
    var = sum_sq_val / group_elems_f32 - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write back
    for offset in range(0, group_elems, BLOCK_SIZE):
        offs = offset + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_elems

        c_idx = offs // (H * W)
        hw = offs % (H * W)
        h_idx = hw // W
        w_idx = hw % W

        ptrs = base_ptr + c_idx * stride_c + h_idx * stride_h + w_idx * stride_w
        x = tl.load(ptrs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        gelu_x = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * inv_sqrt2))
        norm = (gelu_x - mean) * inv_std

        # Channel indices in [0, C)
        c = c_start + c_idx

        w_ptrs = weight_ptr + c
        b_ptrs = bias_ptr + c

        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)

        y = norm * w + b
        y_cast = y.to(x.dtype)

        tl.store(ptrs, y_cast, mask=mask)


def fused_gelu_groupnorm(x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: torch.Tensor,
                         num_groups: int,
                         eps: float = 1e-5) -> torch.Tensor:
    """
    Fused GELU + GroupNorm using Triton.

    Args:
        x:      [N, C, H, W] tensor (CUDA)
        weight: [C] GroupNorm weight
        bias:   [C] GroupNorm bias
        num_groups: number of groups for GroupNorm
        eps:    epsilon for numerical stability

    Returns:
        Tensor of same shape as x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert weight is not None and bias is not None
    assert x.dim() == 4, "Expected NCHW input"
    N, C, H, W = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    assert weight.shape[0] == C and bias.shape[0] == C

    # Ensure contiguous layout (N, C, H, W)
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    stride_n, stride_c, stride_h, stride_w = x.stride()

    # Launch one program per (n, group)
    grid = (N * num_groups,)

    # BLOCK_SIZE must be a power of 2
    gelu_groupnorm_kernel[grid](
        x, weight, bias,
        N, C, H, W,
        num_groups,
        eps,
        stride_n, stride_c, stride_h, stride_w,
        BLOCK_SIZE=256,
    )
    return x


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + fused GELU + GroupNorm (Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose2d in PyTorch (indexing is complex)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
        )
        # Standard GroupNorm module to own parameters (weight, bias, eps, num_groups)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        # PyTorch ConvTranspose2d
        x = self.conv_transpose(x)
        # Fused GELU + GroupNorm using Triton
        x = fused_gelu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        return x
