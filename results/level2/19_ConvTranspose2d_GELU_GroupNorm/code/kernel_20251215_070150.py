# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr,          # *f32 or *f16 / *bf16: [N, C, H, W] (contiguous NCHW)
    weight_ptr,     # *f32 or *f16 / *bf16: [C]
    bias_ptr,       # *f32 or *f16 / *bf16: [C]
    N, C, H, W,
    NUM_GROUPS,         # int
    CHANNELS_PER_GROUP, # int = C // NUM_GROUPS
    eps,                # float
    BLOCK_HW: tl.constexpr,
):
    """
    Fused GELU + GroupNorm over [C, H, W] per sample, per group.

    This kernel is optimized for:
      - Contiguous NCHW layout (x is .contiguous())
      - Minimal integer math in the hot loop (no per-element div/mod)
      - Minimal weight/bias traffic (load once per channel per pass)
      - Coalesced memory access along the spatial (H*W) dimension
    """
    pid = tl.program_id(axis=0)
    groups = NUM_GROUPS

    # Map program id to (n, g)
    n = pid // groups
    g = pid % groups

    if n >= N:
        return

    HW = H * W
    channels_per_group = CHANNELS_PER_GROUP

    # Channel range for this group: [c_start, c_start + channels_per_group)
    c_start = g * channels_per_group

    # Base pointer for this (n, group) slice: element (n, c_start, 0, 0)
    # For contiguous NCHW:
    #   offset = ((n * C + c) * H + h) * W + w
    # So base offset for (n, c_start, 0, 0) is:
    #   base_offset = (n * C + c_start) * HW
    base_offset = (n * C + c_start) * HW
    base_ptr = x_ptr + base_offset

    # First pass: compute mean and variance of GELU(x) over the group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865476

    # Iterate over channels in this group
    for c in range(0, channels_per_group):
        channel_base = base_ptr + c * HW

        # Iterate over spatial elements in chunks of BLOCK_HW
        for hw_offset in range(0, HW, BLOCK_HW):
            offs = hw_offset + tl.arange(0, BLOCK_HW)
            mask = offs < HW

            ptrs = channel_base + offs
            x = tl.load(ptrs, mask=mask, other=0.0)

            # Promote to f32 for numerics
            x_f32 = x.to(tl.float32)

            # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            gelu_x = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * inv_sqrt2))

            # Reduce within this program
            sum_val += tl.sum(gelu_x, axis=0)
            sum_sq_val += tl.sum(gelu_x * gelu_x, axis=0)

    group_elems = channels_per_group * HW
    group_elems_f32 = tl.full((), group_elems, dtype=tl.float32)

    mean = sum_val / group_elems_f32
    var = sum_sq_val / group_elems_f32 - mean * mean
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, apply affine, and write back
    for c in range(0, channels_per_group):
        channel_base = base_ptr + c * HW

        # Global channel index in [0, C)
        c_global = c_start + c

        # Load affine parameters once per channel
        w = tl.load(weight_ptr + c_global).to(tl.float32)
        b = tl.load(bias_ptr + c_global).to(tl.float32)

        for hw_offset in range(0, HW, BLOCK_HW):
            offs = hw_offset + tl.arange(0, BLOCK_HW)
            mask = offs < HW

            ptrs = channel_base + offs
            x = tl.load(ptrs, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)

            gelu_x = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 * inv_sqrt2))
            norm = (gelu_x - mean) * inv_std

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
        x:      [N, C, H, W] tensor (CUDA, contiguous NCHW)
        weight: [C] GroupNorm weight
        bias:   [C] GroupNorm bias
        num_groups: number of groups for GroupNorm
        eps:    epsilon for numerical stability

    Returns:
        Tensor of same shape as x (in-place modified).
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert weight is not None and bias is not None
    assert x.dim() == 4, "Expected NCHW input"
    N, C, H, W = x.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    assert weight.shape[0] == C and bias.shape[0] == C

    # Ensure contiguous layout (N, C, H, W) for simplified indexing
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    channels_per_group = C // num_groups

    # Launch one program per (n, group)
    grid = (N * num_groups,)

    # Tuned for Ada (4090): 256 elements per spatial chunk, 4 warps per program
    gelu_groupnorm_kernel[grid](
        x, weight, bias,
        N, C, H, W,
        num_groups,
        channels_per_group,
        eps,
        BLOCK_HW=256,
        num_warps=4,
        num_stages=2,
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
